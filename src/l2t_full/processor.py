import time
import logging
import uuid
from typing import Optional, Tuple, Any, Dict, List
import torch
from sentence_transformers import SentenceTransformer

from src.llm_client import LLMClient, LLMCallStats
from .dataclasses import L2TFullConfig, L2TGraph, L2TNode, L2TResult
from .enums import L2TNodeCategory, L2TTerminationReason
from .prompt_generator import L2TPromptGenerator as L2TFullPromptGenerator
from .response_parser import L2TResponseParser
from .gnn_module import GNNReasoningActor
from .rl_module import CriticValueNetwork, PPOAgent
from .constants import (
    REWARD_FINAL_ANSWER, REWARD_CONTINUE_SUCCESSFUL_GENERATION,
    REWARD_CONTINUE_FAILED_GENERATION, REWARD_TERMINATE_BRANCH,
    REWARD_BACKTRACK, REWARD_UNCATEGORIZED_FAILURE,
    TEMP_ADJUSTMENT_SCALE, MIN_TEMP, MAX_TEMP,
    NUM_THOUGHTS_ADJUSTMENT_SCALE, MIN_NUM_THOUGHTS, MAX_NUM_THOUGHTS,
    DEFAULT_BASE_TOP_P, TOP_P_ADJUSTMENT_SCALE, MIN_TOP_P, MAX_TOP_P # Added Top P constants
)

logger = logging.getLogger(__name__)

class L2TFullProcessor:
    def __init__(
        self,
        api_key: str,
        config: Optional[L2TFullConfig] = None,
        enable_rate_limiting: bool = True,
        enable_audit_logging: bool = False,
    ):
        self.llm_client = LLMClient(api_key=api_key, enable_rate_limiting=enable_rate_limiting, enable_audit_logging=enable_audit_logging)
        self.config: L2TFullConfig = config or L2TFullConfig()
        self.prompt_generator = L2TFullPromptGenerator(self.config)
        self.logger = logger

        if self.config.gnn_input_dim != 384:
            self.logger.warning(f"Config gnn_input_dim ({self.config.gnn_input_dim}) != 384 (all-MiniLM-L6-v2).")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer: {e}.", exc_info=True)
            self.sentence_model = None

        self.gnn_actor: GNNReasoningActor = GNNReasoningActor(
            input_node_dim=self.config.gnn_input_dim, gcn_hidden_dim=self.config.gcn_hidden_dim,
            mlp_hidden_dim=self.config.mlp_hidden_dim, output_action_dim=self.config.action_dim,
        )
        self.critic_network: CriticValueNetwork = CriticValueNetwork(
            input_dim=self.config.critic_input_dim, hidden_dim=self.config.critic_hidden_dim,
        )
        self.ppo_agent = PPOAgent(
            actor=self.gnn_actor, critic=self.critic_network,
            learning_rate=self.config.ppo_learning_rate, gamma=self.config.ppo_gamma,
            epsilon=self.config.ppo_epsilon, gae_lambda=self.config.ppo_gae_lambda,
            ppo_epochs=self.config.ppo_epochs, batch_size=self.config.ppo_batch_size,
            entropy_coefficient=self.config.ppo_entropy_coefficient,
        )
        self.logger.info("L2TFullProcessor initialized with PyTorch GNN/RL modules and SentenceTransformer.")

    def _update_result_stats(self, result: L2TResult, stats: LLMCallStats):
        result.total_llm_calls += 1
        if stats.completion_tokens is not None: result.total_completion_tokens += stats.completion_tokens
        if stats.prompt_tokens is not None: result.total_prompt_tokens += stats.prompt_tokens
        if stats.call_duration_seconds is not None:
            result.total_llm_interaction_time_seconds += stats.call_duration_seconds

    def _featurize_graph_for_gnn(self, graph: L2TGraph, current_node_id: Optional[str]) -> torch.Tensor:
        if self.sentence_model is None:
            self.logger.error("SentenceTransformer model not available. Returning zero tensor.")
            return torch.zeros(1, self.config.gnn_input_dim)
        if not graph.nodes or current_node_id is None:
            return torch.zeros(1, self.config.gnn_input_dim)
        center_node = graph.get_node(current_node_id)
        if not center_node: return torch.zeros(1, self.config.gnn_input_dim)

        beta = self.config.gnn_subgraph_beta
        subgraph_nodes_map: Dict[str, L2TNode] = {current_node_id: center_node}
        queue: List[Tuple[str, int]] = [(current_node_id, 0)]; head = 0
        while head < len(queue):
            nid, depth = queue[head]; head += 1
            if depth >= beta: continue
            node_obj = subgraph_nodes_map.get(nid)
            if not node_obj: continue
            neighbors_to_add = ([node_obj.parent_id] if node_obj.parent_id else []) + node_obj.children_ids
            for neighbor_id in neighbors_to_add:
                if neighbor_id and neighbor_id not in subgraph_nodes_map:
                    neighbor_node = graph.get_node(neighbor_id)
                    if neighbor_node:
                        subgraph_nodes_map[neighbor_id] = neighbor_node
                        if depth + 1 < beta: queue.append((neighbor_id, depth + 1))

        valid_subgraph_nodes = list(subgraph_nodes_map.values())
        if not valid_subgraph_nodes: return torch.zeros(1, self.config.gnn_input_dim)
        node_contents = [node.content for node in valid_subgraph_nodes]
        try:
            embeddings = self.sentence_model.encode(node_contents, convert_to_tensor=True)
            if embeddings.shape[1] != self.config.gnn_input_dim: # Basic dimension check
                 self.logger.warning(f"Embedding dim {embeddings.shape[1]} != gnn_input_dim {self.config.gnn_input_dim}. Adjusting.")
                 if embeddings.shape[1] > self.config.gnn_input_dim:
                     embeddings = embeddings[:, :self.config.gnn_input_dim]
                 else:
                     padding = torch.zeros(embeddings.shape[0], self.config.gnn_input_dim - embeddings.shape[1], device=embeddings.device)
                     embeddings = torch.cat([embeddings, padding], dim=1)
            aggregated_features = torch.mean(embeddings, dim=0, keepdim=True)
            return aggregated_features
        except Exception as e:
            self.logger.error(f"Error encoding/aggregating node contents: {e}", exc_info=True)
            return torch.zeros(1, self.config.gnn_input_dim)

    def _interpret_action_params(self, action_tensor: torch.Tensor) -> Dict[str, Any]:
        interpreted_params: Dict[str, Any] = {}
        action_dim = self.config.action_dim # Should be 3 based on current setup

        # Ensure action_tensor is 1D for easier indexing
        action_tensor = action_tensor.detach().cpu().squeeze() # Squeeze removes batch dim if it's 1

        self.logger.info(f"Interpreting GNN action_tensor: {action_tensor} (shape: {action_tensor.shape})")

        # Defaults
        interpreted_params['temperature'] = self.config.thought_generation_temperature
        interpreted_params['num_thoughts'] = 1
        interpreted_params['top_p'] = DEFAULT_BASE_TOP_P

        try:
            if action_dim > 0: # Temperature adjustment
                val = action_tensor[0].item()
                temp_adjustment = val * TEMP_ADJUSTMENT_SCALE
                interpreted_params['temperature'] = max(MIN_TEMP, min(self.config.thought_generation_temperature + temp_adjustment, MAX_TEMP))
        except Exception as e: self.logger.warning(f"Error interpreting temperature from action_tensor[0]: {action_tensor}. Error: {e}.")

        try: # Num_thoughts adjustment (conceptual)
            if action_dim > 1:
                val = action_tensor[1].item()
                num_thoughts_adj = round(val * NUM_THOUGHTS_ADJUSTMENT_SCALE) # val could be positive or negative
                interpreted_params['num_thoughts'] = max(MIN_NUM_THOUGHTS, min(MIN_NUM_THOUGHTS + num_thoughts_adj, MAX_NUM_THOUGHTS))
        except Exception as e: self.logger.warning(f"Error interpreting num_thoughts from action_tensor[1]: {action_tensor}. Error: {e}.")

        try: # Top_p adjustment
            if action_dim > 2:
                val = action_tensor[2].item()
                top_p_adjustment = val * TOP_P_ADJUSTMENT_SCALE
                interpreted_params['top_p'] = max(MIN_TOP_P, min(DEFAULT_BASE_TOP_P + top_p_adjustment, MAX_TOP_P))
        except Exception as e: self.logger.warning(f"Error interpreting top_p from action_tensor[2]: {action_tensor}. Error: {e}.")

        self.logger.info(f"Interpreted GNN action params: {interpreted_params}")
        return interpreted_params

    def _calculate_reward(
        self, node_category: L2TNodeCategory, graph: L2TGraph,
        processed_node: L2TNode, result: L2TResult, new_node_content: Optional[str] = None
    ) -> float:
        # Logic remains the same as previous correct version
        reward = 0.0
        if node_category == L2TNodeCategory.FINAL_ANSWER: reward = REWARD_FINAL_ANSWER
        elif node_category == L2TNodeCategory.CONTINUE:
            if new_node_content: reward = REWARD_CONTINUE_SUCCESSFUL_GENERATION
            else: reward = REWARD_CONTINUE_FAILED_GENERATION
        elif node_category == L2TNodeCategory.TERMINATE_BRANCH: reward = REWARD_TERMINATE_BRANCH
        elif node_category == L2TNodeCategory.BACKTRACK: reward = REWARD_BACKTRACK
        else: reward = REWARD_UNCATEGORIZED_FAILURE
        self.logger.info(f"Calculated reward: {reward} for node {processed_node.id}, category {node_category}, new_content: {new_node_content is not None}")
        return reward

    def _process_node(
        self, node_id_to_classify: str, graph: L2TGraph,
        result: L2TResult, current_process_step: int,
    ):
        node_to_classify = graph.get_node(node_id_to_classify)
        if not node_to_classify: self.logger.warning(f"Node {node_id_to_classify} not found. Skipping."); graph.move_to_v_hist(node_id_to_classify); return
        if node_to_classify.category not in [None, L2TNodeCategory.UNCATEGORIZED]: # Check if already classified
            if node_id_to_classify in graph.v_pres: graph.move_to_v_hist(node_id_to_classify)
            return

        self.logger.info(f"Processing node: {node_to_classify.id} ('{node_to_classify.content[:50]}...') at step {current_process_step}")
        current_gnn_input_tensor = self._featurize_graph_for_gnn(graph, node_id_to_classify)

        self.gnn_actor.eval()
        with torch.no_grad():
            action_dist, _ = self.ppo_agent.actor(current_gnn_input_tensor)
            sampled_action_tensor = action_dist.sample()
            log_prob_of_action = action_dist.log_prob(sampled_action_tensor).sum(dim=-1, keepdim=True)

        action_params_interpreted = self._interpret_action_params(sampled_action_tensor)

        graph_context = graph.get_graph_context_for_node_classification(
            node_id_to_classify, self.config.max_prompt_node_references,
            self.config.max_prompt_node_context_tokens, self.config.max_sibling_thoughts_to_include,
        )
        classification_prompt_str = self.prompt_generator.construct_l2t_node_classification_prompt(
                node_to_classify=node_to_classify, graph_context=graph_context, problem_statement=result.problem,
            )
        classification_prompt_obj = [{"role": "user", "content": classification_prompt_str}]

        # Use GNN-derived params for classification call
        classification_llm_temp = action_params_interpreted.get('temperature', self.config.classification_temperature)
        classification_llm_top_p = action_params_interpreted.get('top_p', DEFAULT_BASE_TOP_P) # Use default if not set by GNN

        classification_response, classification_stats = self.llm_client.call(
            models=self.config.classification_model_names, prompt=classification_prompt_obj,
            max_tokens=self.config.classification_max_tokens, temperature=classification_llm_temp, top_p=classification_llm_top_p
        )
        self._update_result_stats(result, classification_stats)
        parsed_classification = L2TResponseParser.parse_l2t_node_classification_response(classification_response)
        node_category = parsed_classification if parsed_classification else L2TNodeCategory.UNCATEGORIZED

        if node_category == L2TNodeCategory.UNCATEGORIZED:
            self.logger.warning(f"Failed to parse classification for node {node_id_to_classify}. Defaulting to TERMINATE_BRANCH.")
            node_category = L2TNodeCategory.TERMINATE_BRANCH

        graph.classify_node(node_id_to_classify, node_category)
        node_to_classify.classification_prompt = classification_prompt_obj
        node_to_classify.classification_response = classification_response

        new_node_content_for_reward: Optional[str] = None
        next_node_for_featurization_id: Optional[str] = node_id_to_classify

        if node_category == L2TNodeCategory.CONTINUE:
            # Use GNN-derived params for thought generation call
            generation_llm_temp = action_params_interpreted.get('temperature', self.config.thought_generation_temperature)
            generation_llm_top_p = action_params_interpreted.get('top_p', DEFAULT_BASE_TOP_P)
            # num_thoughts_to_generate = action_params_interpreted.get('num_thoughts', 1) # Currently not used to gen multiple

            next_thought_prompt_str = self.prompt_generator.construct_l2t_thought_generation_prompt(
                    current_node=node_to_classify, problem_statement=result.problem, graph_context=graph_context,
                )
            next_thought_prompt_obj = [{"role": "user", "content": next_thought_prompt_str}]
            next_thought_response, next_thought_stats = self.llm_client.call(
                models=self.config.thought_generation_model_names, prompt=next_thought_prompt_obj,
                max_tokens=self.config.generation_max_tokens, temperature=generation_llm_temp, top_p=generation_llm_top_p
            )
            self._update_result_stats(result, next_thought_stats)
            parsed_new_thought_content = L2TResponseParser.parse_l2t_thought_generation_response(next_thought_response)

            if parsed_new_thought_content:
                new_node_id = str(uuid.uuid4()); new_node = L2TNode(
                    id=new_node_id, content=parsed_new_thought_content, parent_id=node_id_to_classify,
                    generation_step=current_process_step, prompt_messages=next_thought_prompt_obj,
                    llm_response=next_thought_response, category=L2TNodeCategory.UNCATEGORIZED)
                graph.add_node(new_node); new_node_content_for_reward = new_node.content
                next_node_for_featurization_id = new_node.id; self.logger.info(f"Generated new thought node {new_node_id}")
            else: self.logger.warning(f"Failed to parse new thought from node {node_id_to_classify}.")

        elif node_category == L2TNodeCategory.FINAL_ANSWER:
            result.final_answer = node_to_classify.content; result.succeeded = True
            result.solution_node_id = node_id_to_classify
        elif node_category == L2TNodeCategory.BACKTRACK:
            parent_node = graph.get_parent(node_id_to_classify)
            if parent_node:
                if parent_node.id not in graph.v_hist or parent_node.category == L2TNodeCategory.UNCATEGORIZED:
                    graph.re_add_to_v_pres(parent_node.id); next_node_for_featurization_id = parent_node.id
                    self.logger.info(f"Backtracking to parent node {parent_node.id}.")
                else: self.logger.info(f"Parent {parent_node.id} already processed.")
            else: self.logger.warning(f"Backtrack on node {node_id_to_classify} with no parent.")
        elif node_category == L2TNodeCategory.TERMINATE_BRANCH: self.logger.info(f"Node {node_id_to_classify} classified as TERMINATE_BRANCH.")

        reward = self._calculate_reward(node_category, graph, node_to_classify, result, new_node_content_for_reward)
        next_gnn_input_tensor = self._featurize_graph_for_gnn(graph, next_node_for_featurization_id)
        done = (result.final_answer is not None) or (current_process_step >= self.config.max_steps) or \
               (len(graph.nodes) >= self.config.max_total_nodes)
        self.ppo_agent.store_experience(
            state=current_gnn_input_tensor, action=sampled_action_tensor, old_log_prob=log_prob_of_action,
            reward=reward, next_state=next_gnn_input_tensor, done=done)
        graph.move_to_v_hist(node_id_to_classify)

    def run(self, problem_text: str) -> L2TResult: # Main run loop, mostly unchanged
        start_time = time.time()
        result = L2TResult(problem=problem_text, start_time=start_time)
        graph = L2TGraph(); current_process_step = 0
        initial_prompt_str = self.prompt_generator.construct_l2t_initial_prompt(problem_statement=problem_text)
        initial_prompt_obj = [{"role": "user", "content": initial_prompt_str}]
        initial_response, initial_stats = self.llm_client.call(
            models=self.config.initial_prompt_model_names, prompt=initial_prompt_obj,
            max_tokens=self.config.initial_thought_max_tokens, temperature=self.config.initial_prompt_temperature,
            top_p=DEFAULT_BASE_TOP_P # Initial call uses default top_p
        )
        self._update_result_stats(result, initial_stats)
        parsed_initial_thought_content = L2TResponseParser.parse_l2t_initial_response(initial_response)

        if not parsed_initial_thought_content:
            result.error_message = "Failed to generate initial thought."; result.succeeded = False
            result.termination_reason = L2TTerminationReason.INITIAL_THOUGHT_FAILED
            result.end_time = time.time(); result.total_processing_time_seconds = result.end_time - start_time
            return result

        root_node_id = str(uuid.uuid4())
        root_node = L2TNode(id=root_node_id, content=parsed_initial_thought_content, parent_id=None,
                            generation_step=current_process_step, prompt_messages=initial_prompt_obj,
                            llm_response=initial_response, category=L2TNodeCategory.UNCATEGORIZED)
        graph.add_node(root_node, is_root=True)
        self.logger.info(f"Generated initial thought node {root_node_id}: '{root_node.content[:100]}...'")

        while True:
            current_process_step += 1; self.logger.info(f"Step {current_process_step}, v_pres: {len(graph.v_pres)}")
            if current_process_step > self.config.max_steps: result.termination_reason = L2TTerminationReason.MAX_STEPS_REACHED; break
            if not graph.v_pres: result.termination_reason = L2TTerminationReason.NO_NODES_TO_PROCESS; break
            if (time.time() - start_time) > self.config.max_time_seconds: result.termination_reason = L2TTerminationReason.TIMEOUT; break
            if len(graph.nodes) >= self.config.max_total_nodes: result.termination_reason = L2TTerminationReason.MAX_NODES_REACHED; break
            if result.final_answer is not None: result.termination_reason = L2TTerminationReason.SOLUTION_FOUND; break

            nodes_to_process_this_step = list(graph.v_pres)
            for node_id_to_classify in nodes_to_process_this_step:
                if node_id_to_classify not in graph.v_pres: continue
                node_obj = graph.get_node(node_id_to_classify)
                if not node_obj:
                    self.logger.warning(f"Node {node_id_to_classify} Sched but not found. Removing."); graph.v_pres.remove(node_id_to_classify)
                    if node_id_to_classify not in graph.v_hist: graph.v_hist.append(node_id_to_classify)
                    continue
                self._process_node(node_id_to_classify, graph, result, current_process_step)
                if result.final_answer is not None: break
            if result.final_answer is not None: result.termination_reason = L2TTerminationReason.SOLUTION_FOUND; break

        result.end_time = time.time(); result.total_processing_time_seconds = result.end_time - start_time
        result.reasoning_graph = graph; result.total_nodes = len(graph.nodes); result.total_steps = current_process_step
        if result.final_answer is None and result.succeeded is None:
            result.succeeded = False
            if not result.termination_reason: result.termination_reason = L2TTerminationReason.UNKNOWN

        self.logger.info(f"L2T-Full processing finished. Reason: {result.termination_reason}. Success: {result.succeeded}")
        if self.ppo_agent.experience_buffer:
            self.logger.info("Calling PPOAgent.update() at the end of the run.")
            self.ppo_agent.update()
        else: self.logger.info("PPOAgent experience buffer is empty. Skipping update.")
        return result

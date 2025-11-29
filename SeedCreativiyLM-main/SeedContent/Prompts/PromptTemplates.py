"""
Centralized prompt template management for SeedContent module.
All prompts used across generators and evaluators are defined here for easy modification and maintenance.
"""


class PromptTemplates:
    """
    A centralized class for managing all prompt templates used in the SeedContent module.
    """

    # =============================================================================
    # Answer Generator Prompts (G_Answer)
    # =============================================================================

    """
    CRITICAL: These seeds are for YOUR internal inspiration only. They should influence the MOOD, IMAGERY, or DIRECTION of your writing - but NEVER appear in the actual text. Think of them as the feeling behind your words, not the words themselves. As with all ideas, some may be useful, some may not be immediately relevant, and some seemingly unrelated ones might connect in unexpected, sensory ways.
            You can drop out them as many as you want if you think it's better for the flow.
    
    """

    @staticmethod
    def get_answer_first_stage_prompt(seed_str: str, question: str, recommended_max_tokens: int) -> str:
        return f"""Task: Provide a comprehensive answer to the topic below. This is the first part of a multi-stage response - you will continue in subsequent stages.

        [Thinking Tools]
        The following concepts are provided to help structure your response.
        Use the abstract attributes of these concepts (not just the words) to broaden the perspective or add depth:
        {seed_str}
        
        [Instructions]
        1. Maintain a neutral, objective, and informative tone.
        2. Do not force the keywords into the text. Use them naturally to connect ideas.
        3. Start directly with the core answer.

        Topic:
        {question}

        Your response, Start logically within {recommended_max_tokens} tokens:
        """

    @staticmethod
    def get_answer_middle_stage_prompt(seed_str: str, question: str, recommended_max_tokens: int, previous_content) -> str:
        return f"""Task: Continue the explanation seamlessly. This is a middle part of a multi-stage response - you will continue in subsequent stages.

        [Constraint]
        **DO NOT REPEAT** the preceding text.

        [New Thinking Tools]
        Incorporate attributes of these new concepts to expand the logic or introduce a new facet:
        {seed_str}
        
        [Previous Content]
        {previous_content} ...
        
        [Instructions]
        1. Ensure a perfect grammatical and logical connection to the last word above.
        2. Focus on the flow of information. The seeds should serve the content, not dominate it.
        3. Avoid abrupt changes in style or tone.
        
        Topic (for reference):
        {question}
                        
        Your response, Expand naturally within {recommended_max_tokens} tokens:
        """

    @staticmethod
    def get_answer_last_stage_prompt(seed_str: str, question: str, recommended_max_tokens: int, previous_content) -> str:
        return f"""Task: Conclude the response effectively. This is the final part of a multi-stage answer - bring everything together.

        [Constraint]
        **DO NOT REPEAT** the preceding text.

        [Final Thinking Tools]
        Use these concepts to help synthesize the key points or frame the conclusion:
        {seed_str}
        
        [Previous Content]
        {previous_content} ...

        [Instructions]
        1. Bring the explanation to a clear and satisfying close.
        2. Ensure the conclusion is relevant to the original topic.
        
        Topic (for reference):
        {question}

        Your conclusion, Finish within {recommended_max_tokens} tokens:
        """

    # =============================================================================
    # Core Seed Generator Prompts (SG_Core)
    # =============================================================================

    @staticmethod
    def get_core_seed_prompt(question: str, context_info: str, num_seeds: int) -> str:
        return f"""You are a creative seed generator for exploring diverse conceptual spaces.
                Generate {num_seeds} core seed words that offer unique perspectives on the given context.

                These seeds should be:
                - Single, evocative words
                - Conceptually diverse from each other
                - Either directly related to the topic OR sensorially/metaphorically connected
                - Capable of inspiring creative directions

                Context: {question}{context_info}

                Generate seeds that span different conceptual dimensions - some concrete, some abstract, some metaphorical."""

    # =============================================================================
    # Interpolation Seed Generator Prompts (SG_Interpolation)
    # =============================================================================

    @staticmethod
    def get_interpolation_seed_prompt(core_seeds: list, num_seeds: int) -> str:
        return f"""
        You are generating bridge concepts between existing seed words.

        Core seeds: {', '.join(core_seeds)}

        Generate {num_seeds} interpolation seeds—words that conceptually connect or bridge between the core seeds.

        These should be stepping stones that help traverse the conceptual space between the core ideas.

        Each interpolation seed should:
        - Connect at least two of the core seeds
        - Be a single, clear word
        - Create meaningful conceptual pathways
        """

    # =============================================================================
    # Descendant Seed Generator Prompts (SG_Descendant)
    # =============================================================================

    @staticmethod
    def get_descendant_seed_prompt(core_seeds: list, interpolation_seeds: list, num_seeds: int) -> str:
        return f"""You are generating deep conceptual extensions from existing seeds.

               Existing seeds:
               Core: {', '.join(core_seeds)}
               Interpolation: {', '.join(interpolation_seeds)}

               Generate {num_seeds} descendant seeds.

               STRICT RULES:
               - Each seed MUST be 1-3 words MAXIMUM (e.g., "Flux", "Silent Echo", "Fading Light")
               - NO explanations, NO parentheses, NO descriptions
               - Just the words themselves, nothing more

               Each descendant seed should explore implicit or underlying dimensions of the existing seeds."""

    # =============================================================================
    # Creativity Evaluation Prompts (E_CreativityEvaluation)
    # =============================================================================

    @staticmethod
    def get_creativity_evaluation_prompt(chosen_story: str, seed_answer: str) -> str:
        return f"""
            You're evaluating creative writing responses A and B.
            Compare them based on these dimensions:
            - Imagery: vivid descriptions and sensory details
            - Tension: dramatic interest and conflict
            - Pattern: structural elements and composition
            - Energy: engaging style and dynamic writing
            - Insight: meaningful ideas and depth

            IMPORTANT: Your answer MUST use EXACTLY this format:
            Reasoning: [brief comparison]
            Preferred: [A or B]

            Here are the two responses:

            Response A:
            {chosen_story}

            Response B:
            {seed_answer}

            Please provide your evaluation:

            """

    # =============================================================================
    # Control Group Generator Prompts (G_ControlGroup)
    # =============================================================================

    @staticmethod
    def get_control_group_prompt(question: str, recommended_max_tokens: int) -> str:
        return f"""You are creating a creative response to a question.

            Now, generate a comprehensive and detailed continuation of the response to the following:

            {question}

            (The response should be longer than {recommended_max_tokens} tokens)

            Write your response. Be vivid and creative, ensuring continuity with the previous content.
            """

    @torch.no_grad()
    def jacobi(
        self,
        prompt,
        response,
    ):

        prompt_len, resp_len = [
            len(p) for p in prompt
        ], [len(r) for r in response]

        (
            response_input_ids,
            response_attention_mask,
        ) = self.prepare_inputs(prompt)

        (
            prompt_key_values,
            prompt_logits,
            prompt_attention_mask,
        ) = self.get_starting_cache(prompt)

        prompt_start = (
            response_attention_mask.shape[1]
            - response_attention_mask.sum(
                -1
            )
        )

        response_start = (
            prompt_start.cpu()
            + torch.tensor(
                list(prompt_len),
            )
        )

        (
            response_input_ids,
            response_attention_mask,
        ) = self.prepare_inputs(
            [
                resp
                + [
                    self.tokenizer.eos_token_id
                ]
                for resp in response
            ],
            padding_side="right",
        )

        response_outputs = self.model.forward(
            input_ids=response_input_ids,
            attention_mask=torch.cat(
                [
                    prompt_attention_mask,
                    response_attention_mask,
                ],
                dim=-1,
            ),
            return_dict=True,
            use_cache=True,
            past_key_values=prompt_key_values,
        )

        new_response = sample_jacobi(
            logits=torch.cat(
                [
                    prompt_logits[
                        :, :-1, :
                    ],
                    response_outputs.logits,
                ],
                dim=-2,
            ),
            response_start=[1]
            * len(prompt),
            response_length=resp_len,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        (
            new_response_input_ids,
            new_response_attention_mask,
        ) = self.prepare_inputs(
            [
                resp
                + [
                    self.tokenizer.eos_token_id
                ]
                for resp in new_response
            ],
            padding_side="right",
        )

        new_response_outputs = self.model.forward(
            input_ids=new_response_input_ids,
            attention_mask=torch.cat(
                [
                    prompt_attention_mask,
                    new_response_attention_mask,
                ],
                dim=-1,
            ),
            return_dict=True,
            use_cache=True,
            past_key_values=prompt_key_values,
        )

        import pdb

        pdb.set_trace()

        response_logprobs = log_softmax(
            response_outputs.logits, dim=-1
        )

        response_likelihood = transition_scores(
            logprobs=response_logprobs,
            response_start=response_start,
            response=response,
        )

        forward_transition_scores = transition_scores(
            logprobs=response_logprobs,
            response_start=response_start,
            response=new_response,
        )

        (
            response_input_ids,
            response_attention_mask,
        ) = self.prepare_inputs(
            [
                resp
                + [
                    self.tokenizer.eos_token_id
                ]
                for resp in response
            ],
            padding_side="right",
        )

        (
            new_response_input_ids,
            new_response_attention_mask,
        ) = self.prepare_inputs(
            [
                resp
                + [
                    self.tokenizer.eos_token_id
                ]
                for resp in (new_response)
            ],
            padding_side="right",
        )

        new_response_outputs = self.model.forward(
            input_ids=response_input_ids,
            attention_mask=new_response_attention_mask,
            return_dict=True,
            use_cache=True,
            past_key_values=past_key_values,
        )

        new_response_logprobs = log_softmax(
            new_response_outputs.logits,
            dim=-1,
        )

        new_response_likelihood = transition_scores(
            logprobs=new_response_logprobs,
            response_start=response_start,
            response=new_response,
        )

        backward_transition_scores = transition_scores(
            logprobs=new_response_logprobs,
            response_start=response_start,
            response=response,
        )

        ###

        # forward_transition_scores, backward_transition_scores
        # response_likelihood, new_response_likelihood

        # response, new_response

        ###

        import pdb

        pdb.set_trace()
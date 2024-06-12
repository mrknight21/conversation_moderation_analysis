annotators ={
    "anh": 0,
    "anudeex": 1,
    "cam": 2,
    "claudia": 3,
    "jemima": 4
}

dialogue_acts ={"Probing": 0, "Confronting": 1, "Instruction": 2, "Interpretation": 3, "Supplement": 4,
              "All Utility": 5}

attributes = {

    "informational motive": {
        "type": "binary",
        "definition": "This motive centers on the exchange of relevant knowledge and information in a conversation, with a focus on both imparting and acquiring details that are pertinent to the topic or goal of the discussion. It encompasses the sharing of opinions, rationales, and specific information that contribute constructively to the dialogue. ",
        "examples": '“Why do you think minimum wage is unfair?” (Relevant information seeking.) “The legal system has many loopholes.” (Expressing opinion.) “Yea! I agree with your point!” (Agreement relevant to the topic.)  “The law was established in 1998.” (Providing topic relevant information.)'
    },

    "social motive": {
        "type": "binary",
        "definition": "This motive aims to enhance the social and emotional atmosphere of a discussion, ensuring that participants are engaged and motivated, and that their relationships are fostered. It includes responses designed to influence the feelings, emotions, and interpersonal dynamics among the participants, contributing positively to the collective experience of the conversation. ",
        "examples": '“It is sad to hear the news of the tragedy.” (Expressing emotion and feeling.) “Thank you! Mr. Wang.” (Appreciating.) “Hello! Let’s welcome Dr. Frankton.” (Greeting.) “I can understand your struggle being a single mum.” (Empathy) “How do you feel? when your work was totally denied.” (Exploring other’s feeling.) “Please feel free to say your mind because I can’t bite you online, hehe!” (Humour.) “The definition is short and simple! I love it!” (Encouragement.) “Maybe Amy’s intention is different to what you thought, you guys actually believe the same thing.” (Social Reframing.)'
    },

    "coordinative motive": {
        "type": "binary",
        "definition": "This motive focuses on ensuring the smooth flow of the dialogue by adhering to established rules, plans, and the overall context in which the conversation takes place. It involves guiding the interaction to maintain order and efficiency, facilitating a structured and coherent exchange of ideas among participants. ",
        "examples": "“Let’s move on to the next question due to time running out.” (Command) “We going to start with the blue team and then the red team” (Planning) “Do you want to go first?” (Asking for process preference.) “Please move to the left side and turn on your mic!” (Managing environment) "
    },


    "dialogue act": {
        "type": "multiclass",
        "instruction":"Your role is an annotator, annotating the moderation behavior and speech of a debate TV show. The debate topic is 'Gun Reduces Crime', given the definition and the examples, the context of prior dialogue, please label which dialogue act category does the target sentence belong to?",
        "options": {
            "Probing":{
                "definition":"Response that prompts speaker for contributing information (this excludes rhetorical question).",
                "examples": '“What is your view on that Dr. Foster?” (Questioning.) “Where are you from?” (Social questioning.) “Peter!” (Name calling for response.) “If the majority of people are voting against it, would you still insist?” (Elaborated questioning.) “Do you agree with this statement?” (Binary question.)'
            },
            "Confronting":{
                "definition":"Response that prompts one speaker to response or engage with another speaker(s)'s statements, questions or opinions.",
                "examples": '“So David pointed out the critical weakness of the system, what is your thought on his critiques, Dr. Foster?”, "Judge Anderson, what is your response to this hypothetical scenario posed by Ms. Lee regarding privacy laws?", "Senator Harris, you have proposed reducing taxes instead. How do you respond to Mr. Walkers suggestion to increase school funding?", "So, Dr. Green, Professor Brown just criticized the emissions policy. What is your response to his critique?"'
            },
            "Supplement": {
                "definition":"Responses categorized as supplements provide additional information by explaining, expressing feelings, proposals, hypotheses, opinions, or introducing external knowledge. These responses enrich the conversation by bringing in new elements or details, helping to expand the context or understanding of the discussion without directly influencing immediate actions.",
                "examples": '“And that concludes round one of this Intelligence Squared U.S. debate where our motion is Break up the Big Banks.” (Addressing progess) “The blue team will go first, then the red team can speak” (explaining program rule) “Supposed we live in a world where such behaviour is accepted.” (Hypothesis) “I suggest the best solution is giving everyone equal chances.” (Proposal) “The government announced tax raise from March.” (Providing external information) “I agree with that you said.” (Agreement) “GM means genetic modified.” (Providing external knowledge) “I think people should be given the right to say no!” (Opinion)'
            },
            "Interpretation": {
                "definition":"Interpretation responses help to clarify, reframe, summarize, paraphrase, or link back to previous statements within the dialogue. These responses are used to elucidate the meaning of earlier exchanges, often to ensure understanding, provide clarity, or connect different parts of the conversation.",
                "examples": '“So basically, what Amy said is that they didn’t use the budget efficiently”. (Summarisation) “You said ‘I believe GM is harmless,’.” (Quote) “In another word, you don’t like their plan.”. (Paraphrase) “My understanding is you don’t support this due to moral reason.” (Interpretation) “She does not mean to hurt you but just tell the truth.” (Clarify) “So far, we have Dr. Johnson suggesting…., and Dr. Brown against it because……”(Summarisation) “Amy saying that to justify the reduction of the wage, but not aiming to induce suffering.” (Reframing) '
            },
            "Instruction": {
                "definition":"Instructions are communications that carry the explicit intent to command, influence, halt, or shape the immediate actions of the recipients. These responses are intended to prompt a specific and immediate change in behavior, adherence to rules, or modification of the dialogue process.",
                "examples": '“Please get back to the topic.” (Commanding) “Please stop here, we are running out of time.” (Reminding of the rule) “The red will start now.” (Instruction) “Please mind your choice of words and manner.” (social policing) “Do not intentionally create misconception.” (argumentative policing) “Now is not your term, stop here.” (coordinative policing) '
            },
            "All Utility": {
                "definition":"All other unspecified acts. Occasionally, this group of behaviours play an important role to show engagement (e.g. back channelling) and getting attention (e.g. floor grabbing).",
                "examples": '“Thanks, you.” (Greeting) “Sorry.” (Apology) “Okay.” (Back channelling) “Um hm.” (Back channelling) “But, but, but……” (Floor grabbing) '
            }
        }
    },

    "target speaker":{
        "type": "multiclass",
        "instruction": "",
        "examples": '“We are going to start in 10 minutes. The red team will go first.” (talking to everyone). “Paul, what is your thought?” (talking to Paul Helmke) “Cough! Cough!” (Self) “The guy sitting at the front row. Yes! You!” (talking to Audience) “This is ‘Intelligence Square’. Welcome back!” (talking to Audience) '
    }

}


def construct_prompt_unit(instance):
    prompt = ""

    topic = instance["meta"]["topic"]
    speakers = instance["meta"]["speakers"]
    target = instance["target"]
    instruction = f'Your role is an annotator, annotating the moderation behavior and speech of a debate TV show. The debate topic is "{topic}", given the definition and the examples, the context of prior dialogue, please label which motives the target response carries? And which dialogue act the target sentence belong to? And who is the moderator talking to? \n\n'

    prompt += instruction

    speakers = ", ".join([f'"{s}"' for s in speakers])

    motive_intro = "Motives: During the dialogue, the moderator is acting upon a mixed-motives scenario, where different motives are expressed through responses depending on the context of the dialogue. Different from dialogue act, motives are the high level goal that the moderator aim to achieve. The definitions and examples of the 3 motives are below: \n\n"

    prompt += motive_intro

    im_def = f'informational motive: {attributes["informational motive"]["definition"]} \n'
    im_ex = f'examples: {attributes["informational motive"]["examples"]} \n\n'
    prompt += im_def
    prompt += im_ex


    sm_def = f'social motive: {attributes["social motive"]["definition"]} \n'
    sm_ex = f'examples: {attributes["social motive"]["examples"]} \n\n'
    prompt += sm_def
    prompt += sm_ex

    cm_def = f'coordinative motive: {attributes["coordinative motive"]["definition"]} \n'
    cm_ex = f'examples: {attributes["coordinative motive"]["examples"]} \n\n'
    prompt += cm_def
    prompt += cm_ex

    dialogue_act_intro = "Dialogue act: Dialogue acts is referring to the function of a piece of a speech. The definitions and examples of the 6 motives are below:"
    prompt += dialogue_act_intro

    pq_def = f'Probing: {attributes["dialogue act"]["options"]["Probing"]["definition"]} \n'
    pq_ex = f'examples: {attributes["dialogue act"]["options"]["Probing"]["examples"]} \n\n'
    prompt += pq_def
    prompt += pq_ex

    cq_def = f'Confronting: {attributes["dialogue act"]["options"]["Confronting"]["definition"]} \n'
    cq_ex = f'examples: {attributes["dialogue act"]["options"]["Confronting"]["examples"]} \n\n'
    prompt += cq_def
    prompt += cq_ex

    sp_def = f'Supplement: {attributes["dialogue act"]["options"]["Supplement"]["definition"]} \n'
    sp_ex = f'examples: {attributes["dialogue act"]["options"]["Supplement"]["examples"]} \n\n'
    prompt += sp_def
    prompt += sp_ex

    it_def = f'Interpretation: {attributes["dialogue act"]["options"]["Interpretation"]["definition"]} \n'
    it_ex = f'examples: {attributes["dialogue act"]["options"]["Interpretation"]["examples"]} \n\n'
    prompt += it_def
    prompt += it_ex

    is_def = f'Instruction: {attributes["dialogue act"]["options"]["Instruction"]["definition"]} \n'
    is_ex = f'examples: {attributes["dialogue act"]["options"]["Instruction"]["examples"]} \n\n'
    prompt += is_def
    prompt += is_ex

    ut_def = f'All Utility: {attributes["dialogue act"]["options"]["All Utility"]["definition"]} \n'
    ut_ex = f'examples: {attributes["dialogue act"]["options"]["All Utility"]["examples"]} \n\n'
    prompt += ut_def
    prompt += ut_ex

    additional_instruction = "Sometime several dialogue act labels might be viable, in this case please follow the steps below to determine priority: \n" \
                             "1. Does the sentence shows intent to elicit more information? if so go to step 2, else go to step 3.\n" \
                             '2. Does the sentence involve, engage, or mention another speaker outside the target speaker? If yes please select "Confronting", else select "Probing".\n' \
                             '3. Does the sentence intend to change or instruct the behavior of the target speaker in the near future turns? If yes please select "Instruction", else go to step 4.\n' \
                             '4. Does the sentence provide some insight, expression or information? If no, please select "All Utility", else go to step 5.\n' \
                             '5. Does the information of the sentence involve information from earlier dialogue? If yes please select "Interpretation", else select "Supplement". \n\n'

    prompt += additional_instruction

    if len(instance["context"]["prior_context"]) > 0:
        prompt += "Dialogue context before the target sentence:\n"
        for s in instance["context"]["prior_context"]:
            prompt += f"{s[0]} ({s[1]}): {s[2]} \n"

    prompt += "\nTarget sentence:\n"
    prompt += f"{target[0]} ({target[1]}): {target[2]} \n"

    if len(instance["context"]["post_context"]) > 0:
        prompt += "\nDialogue context after the target sentence:\n"
        for s in instance["context"]["post_context"]:
            prompt += f"{s[0]} ({s[1]}): {s[2]} \n"

    prompt += "\n"

    prompt += "Please answer with the JSON format:{"
    prompt += '"motives": List(None or more from "informational motive", "social motive", "coordinative motive"),'
    prompt += '"dialogue act": String(one option from "Probing", "Confronting", "Supplement", "Interpretation", "Instruction", "All Utility"),'
    prompt += '"target speaker(s)": String(one option from ' + speakers + '),'
    prompt += '"reason": String'
    prompt += "}\n"
    prompt += "For example: \n"
    prompt += 'answer: {"motive": ["informational motive"], "dialogue act": "Probing",  "target speaker(s)": "7 (Joe Smith- for)", "reason": "The moderator asks a question to Joe Smith aimed at eliciting his viewpoint or reaction to a statement from the recent policy change for combatting climate change......"}'

    return prompt
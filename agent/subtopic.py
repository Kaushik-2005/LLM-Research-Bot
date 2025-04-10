import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from numpy import intp

load_dotenv()

class ResearchSubtopicGenerator:
    STYLE_TYPES = {
        "emerging_trends": "Emerging Trends",
        "academic_categories": "Academic Categories",
        "industrial_applications": "Industrial Applications",
        "few_shot_format": "Few-Shot Format"
    }

    def __init__(self, model_name = "llama-3.3-70b-versatile", temperature = 0.7, verbose = False):
        self.verbose = verbose
        self.llm = ChatGroq(
            temperature = temperature,
            groq_api_key = os.getenv("GROQ_API_KEY"), # type: ignore
            model_name = model_name # type: ignore
        )

        self.prompt = PromptTemplate.from_template(
            """
            You are a specialized research assistant.
            Given a research topic: "{topic}"
            And a style_type: "{style_type}" 

            Where style_type can be one of the following:
                - "emerging_trends" → for cutting-edge and recent research directions
                - "academic_categories" → for typical journal/conference sub-domains
                - "industrial_applications" → for practical and industry-related use cases
                - "few_shot_format" → for few-shot/NLP prompt-friendly format

            Generate a plain numbered list of relevant sub-topics or applications based on the style_type.

            Rules:
                - No markdown or bullet points.
                - Keep each item clear and distinct.
                - Return a maximum of 4-5 items.

            Respond appropriately according to the style_type:

            If style_type is:
                - "emerging_trends": focus on innovations and recently growing areas.
                - "academic_categories": return categories typically found in top academic journals or conferences.
                - "industrial_applications": highlight real-world domains or industries using the research.
                - "few_shot_format": follow this layout:
                    Topic: {topic}
                    Research Areas:
                        1. ...
                        2. ...
                        3. ...
            Begin output.
            """
        )
        self.chain = self.prompt | self.llm

    def generate_all_styles(self, topic):
        all_subtopics = []
        index = 1
        for style_key, style_name in self.STYLE_TYPES.items():
            if self.verbose:
                print(f"Generating for: {style_name}...")
            sub_topics_raw = self.chain.invoke({"topic": topic, "style_type": style_key})
            raw_text = sub_topics_raw.content

            subtopics = []
            for line in raw_text.split("\n"): # type: ignore
                if line.strip() and line.lstrip()[0].isdigit():
                    content = line.split(".", 1)[-1].strip()
                    subtopics.append((index, content, style_name))
                    index += 1
            all_subtopics.extend(subtopics)

        return all_subtopics

    def refine_subtopics(self, all_subtopics):
        while True:
            print("\nGenerated Subtopics:\n")
            for idx, content, style in all_subtopics:
                print(f"{idx}. ({style}) {content}")
            
            user_input = input(
                "\nEnter numbers of subtopics to keep (comma-separated), 'all' to keep all, or 'regen' to regenerate: "
            )

            if user_input.lower() == "regen":
                print("\nRegenerating subtopics...\n")
                return None

            if user_input.lower() == "all":
                return [content for _, content, _ in all_subtopics]
            
            try:
                selected_indices = [int(i.strip()) for i in user_input.split(",")]
                selected_subtopics = [
                    content for idx, content, _ in all_subtopics if idx in selected_indices
                ]

                if selected_subtopics:
                    return selected_subtopics
                else:
                    print("Invalid selection. Please enter valid numbers.")
            except Exception:
                print("Invalid input. Please enter numbers separated by commas.")

    def run(self):
        topic = input("Enter your research topic: ")

        while True:
            all_subtopics = self.generate_all_styles(topic)
            selected_subtopics = self.refine_subtopics(all_subtopics)
            if selected_subtopics is not None:
                break  # Finalized subtopics

        print("\nFinalized Subtopics:")
        for subtopic in selected_subtopics:
            print(f"- {subtopic}")
        return selected_subtopics

if __name__ == "__main__":
    generator = ResearchSubtopicGenerator(verbose = True)
    generator.run()
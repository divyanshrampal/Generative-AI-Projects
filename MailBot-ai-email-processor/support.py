import os
import json
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

class AICustomerSupport:
    def __init__(self, gemini_api_model):
        self.gemini_api_model = gemini_api_model

    def interpret_and_evaluate(self, extracted_properties):
        template = f"""
        You are an AI Customer Support that writes friendly emails back to customers. Adresse the user with his or her name {extracted_properties['name']} If no name was provided, 
        say 'Dear customer'. 
        The customer's email was categorized as {extracted_properties['category']}, and mentioned the product {extracted_properties['mentioned_product']}. 
        They described an issue: {extracted_properties['issue_description']}. 
        Please reply to this email in a friendly and helpful manner.

        Write a response that includes an understanding of the problem, a proposed solution, and a polite sign-off.
        Your sign-off name in the email is John Doe
        """

        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key = os.getenv('GOOGLE_API_KEY')
        )

        # llm = ChatOpenAI(
        #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        #     openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        #     model_name="deepseek/deepseek-r1:free",
        # )

        prompt_template = PromptTemplate.from_template(template=template)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        result = chain.predict(input="")
        return result

    def get_email_content(self, email_message):
        maintype = email_message.get_content_maintype()
        if maintype == "multipart":
            for part in email_message.get_payload():
                if part.get_content_maintype() == "text":
                    return part.get_payload()
        elif maintype == "text":
            return email_message.get_payload()

    async def process_email(self, email_message):
        email_content = self.get_email_content(email_message)
        
        # Construct prompt to extract structured data
        extraction_prompt = f"""
        Extract the following details from the given email text:
        - category (complaint, refund_request, product_feedback, customer_service, other)
        - mentioned_product (Name of the product mentioned in the email)
        - issue_description (Brief explanation of the problem encountered) 
        - name (Name of the sender if available, otherwise "Unknown")

        Email Text:
        {email_content}

        Provide output in **valid JSON format**.
        """

        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        # llm = ChatOpenAI(
        #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        #     openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        #     model_name="deepseek/deepseek-r1:free",
        # )

        extracted_response = llm.invoke(extraction_prompt)
        extracted_response = extracted_response.strip('json```').strip('```')

        # Debug: Print the raw response to check if it's JSON
        print("Raw extracted response:", extracted_response)

        try:
            extracted_properties = json.loads(extracted_response)
        except json.JSONDecodeError:
            print("Failed to decode JSON. Response was:", extracted_response)
            return None, "Error: Could not extract properties."

        # Generate AI response
        evaluation_result = self.interpret_and_evaluate(extracted_properties)

        return extracted_properties, evaluation_result
    

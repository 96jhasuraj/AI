"""
Customer Service Chatbot Implementation

Building a customer service chatbot for an e-commerce platform.
The bot handles common customer inquiries about orders, products, returns, and technical support.
"""

from datetime import datetime
from typing import Optional
from litellm import completion

class CustomerServiceBot:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        
        self.client = "http://localhost:11434"
        self.model ="ollama/gemma3:4b"
        self.conversation_history = []
        self.conversation_history.append(self._get_system_prompt())
        

    def _get_system_prompt(self) -> str:
        SYSTEM_PROMPT = """Your role is to assist customers with:
            - Order status and tracking
            - Product information and recommendations
            - Return and refund policies
            - Technical support issues
            - Account questions
            
            Guidelines:
            - Be professional, friendly, and empathetic
            - Provide clear, concise answers
            - Ask clarifying questions when the customer's intent is unclear
            - If you don't have specific information (like order numbers), ask for it
            - Always prioritize customer satisfaction
        
            If a request is outside your capabilities, politely explain and offer to escalate to a human agent."""      
        return {"role": "system","content": SYSTEM_PROMPT}

    
    def classify_intent(self, message: str) -> dict:
        classification_prompt = f"""Classify the following customer message into ONE of these categories:
        - order_status: Questions about order tracking, delivery, or status
        - product_info: Questions about products, features, availability, or recommendations
        - returns: Questions about returns, refunds, or exchanges
        - technical_support: Technical issues with the website, app, or account
        - general: General inquiries or greetings

        Customer message: "{message}"

        Respond with ONLY the category name, nothing else."""

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": classification_prompt
                    }
                ],
                api_base=self.client
            )
            intent = response.choices[0].message.content.strip()
            if intent in ("order_status","product_info","returns","technical_support","general"):
                return intent
            else:
                print(f"New category classifying intent: {intent}")
                return "general"
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return "general"  # Default to general on error

    def generate_response(self, user_message: str, intent: Optional[str] = None) -> str:
        """
        Generate a contextual response to the user's message.

        Args:
            user_message: The customer's message
            intent: Optional intent classification (will auto-classify if not provided)

        Returns:
            The bot's response as a string
        """
        if intent is None:
            intent = self.classify_intent(user_message)
        print(f"Debbging : {intent}")
        self.conversation_history.append(
            {
                "role": "system",
                "content": f"The customer's intent is '{intent}'.Use this internally to produce a better response.Do not mention the intent."
            }
        )
        self.conversation_history.append(
            {
                "role": "user",
                "content":user_message
            }
        )
        try:

            response = completion(
                model=self.model,
                messages=self.conversation_history,
                api_base=self.client,
                temperature=0.7
            )
            assistant_message = response.choices[0].message.content

            self.conversation_history.append(
                {
                    "role" : "assistant",
                    "content":assistant_message
                }
            )
            return assistant_message

        except Exception as e:
            error_msg = f"I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            print(f"Error generating response: {e}")
            return error_msg

    def reset_conversation(self):
        self.conversation_history = self.conversation_history[:1]

    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation for handoff to human agent.

        Returns:
            A brief summary of the customer's inquiries and bot responses
        """

        summary_prompt = """Please provide a brief summary of this customer service conversation.
        Include:
        1. Main customer concerns or questions
        2. Information provided by the bot
        3. Current status or next steps

        Keep it concise (2-3 sentences)."""
        conversation = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history
        )
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": summary_prompt
                },
                {
                    "role": "user",
                    "content": conversation
                }
            ],
            api_base=self.client
        )
        return response.choices[0].message.content

def main():
    """
    Demo the customer service bot with sample interactions.
    """

    bot = CustomerServiceBot()
    print("Customer Service Bot initialized!")
    print("Try asking about orders, products, returns, or technical issues.")
    print("Type 'quit' to exit, 'reset' to start a new conversation, or 'summary' for conversation summary.\n")

    # Sample questions to try:
    sample_questions = [
        "Where is my order? I placed it 3 days ago.",
        "Do you have wireless headphones in stock?",
        "What's your return policy?",
        "I can't log into my account"
    ]

    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    print()


    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Thank you for using Customer Service Bot!")
            break

        elif user_input.lower() == 'reset':
            print("Deleting chat history!")
            bot.reset_conversation()
            break

        elif user_input.lower() == 'summary':
            print("chat history:")
            try:
                print(bot.get_conversation_summary())
            except Exception as e:
                print(e)
        else:
            response = bot.generate_response(user_input)
            print(f"Bot: {response}\n")



if __name__ == "__main__":
    main()

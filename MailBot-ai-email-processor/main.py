import asyncio
from dotenv import load_dotenv
from extraction import EmailHandler
from support import AICustomerSupport

async def fetch_and_process_emails(fetcher, ai_support):
    while True:
        print("Checking for new emails...")
        new_emails = await fetcher.get_new_emails()
        for email_message, sender_name, sender_addr in new_emails:
            if sender_addr not in fetcher.whitelist:
                print(
                    f"Ignored email from: {sender_name} ({sender_addr}) - Not in whitelist"
                )
                continue

            extracted_properties, evaluation_result = await ai_support.process_email(
                email_message
            )
            print(f"Email from: {sender_name} ({sender_addr})")

            subject = "AI Customer Service Reply"
            fetcher.send_email(sender_addr, subject, evaluation_result)

        print("Sleeping for 10 seconds...")
        await asyncio.sleep(10)

async def main():
    load_dotenv()
    fetcher = EmailHandler()
    ai_support = AICustomerSupport(gemini_api_model="gemini-pro")
    await fetch_and_process_emails(fetcher, ai_support)


if __name__ == "__main__":
    asyncio.run(main())
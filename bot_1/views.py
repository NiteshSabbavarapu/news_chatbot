import requests
import random
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.middleware.csrf import get_token
from django.middleware.common import CommonMiddleware
import json
from groq import Groq
from crewai_tools import SerperDevTool
from django.utils.decorators import decorator_from_middleware
from django.middleware.csrf import CsrfViewMiddleware

# Initialize Groq client using environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize SerperDevTool for internet search
serper_api_key = os.getenv("SERPER_API_KEY")
tool = SerperDevTool(api_key=serper_api_key,
                     search_url="https://google.serper.dev/scholar",
                     n_results=5)

# Store conversation history
conversation_history = []


def fetch_search_results(user_input):
    """Use SerperDevTool to search the web based on user input."""
    try:
        return tool.run(search_query=user_input)
    except Exception as e:
        return {"error": f"Error fetching search results: {str(e)}"}


def rephrase_response(content):
    """Use Groq model to rephrase fetched content for a user-friendly output."""
    if isinstance(content, list):
        content = " ".join(content)  # Convert list to a string if necessary

    prompt = (
            "Summarize the following news content into a structured JSON format. "
            "The response must be in strict JSON format with keys: 'articles', where each article has 'title', 'summary', and 'source'. "
            "Do NOT include any additional text or explanations, only a valid JSON object. "
            "Here is the content:\n" + str(content)
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(content)}
            ],
            temperature=0.8,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        response_text = completion.choices[0].message.content.strip()
        print("Groq API Response:", response_text)  # Debugging

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON format received from AI. Raw response: " + response_text}

    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}


def fetch_groq_response(user_input):
    """Use Groq API to generate a normal conversation response."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": "Respond to the user query in a conversational manner."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.8,
            max_completion_tokens=512,
            top_p=1,
            stream=False
        )

        return {"response": completion.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}


@csrf_exempt
def chatbot_view(request):
    """Django view for handling chatbot interactions."""
    if request.method == "OPTIONS":
        response = JsonResponse({})
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "").lower().strip()
            conversation_history.append(user_input)

            if any(keyword in user_input for keyword in
                   ["news", "update", "latest", "headline", "breaking"]):
                search_results = fetch_search_results(user_input)
                if not search_results or "error" in search_results:
                    return JsonResponse({
                                            "error": "No relevant news updates found. Try another query."})

                bot_response = rephrase_response(search_results)
                if isinstance(bot_response, dict) and "error" in bot_response:
                    return JsonResponse(bot_response, status=500)

                return JsonResponse(bot_response)
            else:
                bot_response = fetch_groq_response(user_input)
                return JsonResponse(bot_response)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request format."},
                                status=400)

    response = JsonResponse({
                                "message": "Welcome! Ask me anything about the latest news updates or have a normal conversation."})
    response["Access-Control-Allow-Origin"] = "*"
    return response

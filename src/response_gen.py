# response_gen.py
from rag_layer import retrieve_context

def generate_response(recommendations, user_profile):
    response = "### Supplement Recommendations Based on Your Profile\n\n"
    response += ("Based on the information you provided (diet: {diet}, health goals: {goals}, "
                 "lifestyle: {lifestyle}, conditions: {conditions}, location: {location}), "
                 "we recommend the following supplements:\n\n").format(
        diet=user_profile.get("diet", "N/A"),
        goals=", ".join(user_profile.get("health_goals", [])),
        lifestyle=user_profile.get("lifestyle", "N/A"),
        conditions=", ".join(user_profile.get("conditions", [])),
        location=user_profile.get("location", "N/A")
    )
    
    for rec in recommendations:
        context = retrieve_context(rec)
        response += f"- **{rec}**: {context}\n"
    
    response += ("\nLocal purchasing options and pricing details can be found on our partner sites. "
                 "(This is a placeholder for local store integration.)\n")
    return response

if __name__ == "__main__":
    # Test response generation
    sample_recs = ["Supplement A", "Supplement B", "Supplement C"]
    dummy_profile = {
        "diet": "Omnivore",
        "health_goals": ["Muscle Gain", "General Wellness"],
        "lifestyle": "Active",
        "conditions": ["None"],
        "location": "New York, USA"
    }
    print(generate_response(sample_recs, dummy_profile))

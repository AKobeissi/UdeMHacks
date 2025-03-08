# rag_layer.py
def retrieve_context(supplement_name):
    # Dummy scientific context for each supplement
    knowledge_base = {
        "Supplement A": "Research shows Supplement A improves energy metabolism by supporting mitochondrial function.",
        "Supplement B": "Clinical studies indicate Supplement B supports muscle growth by increasing protein synthesis.",
        "Supplement C": "Evidence suggests Supplement C aids digestive health by enhancing gut flora balance."
    }
    return knowledge_base.get(supplement_name, "No additional research context available.")

if __name__ == "__main__":
    # Example retrieval test
    context = retrieve_context("Supplement A")
    print("Context for Supplement A:", context)

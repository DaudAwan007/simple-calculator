{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNC69kGiI2Fv9NyJP2SZ84g",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DaudAwan007/simple-calculator/blob/main/chatbot.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install necessary libraries\n",
        "# Uncomment the following lines if running for the first time\n",
        "# !pip install streamlit transformers torch\n",
        "\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "import torch\n",
        "\n",
        "# Load pre-trained DialoGPT model and tokenizer\n",
        "@st.cache_resource\n",
        "def load_model():\n",
        "    model_name = \"microsoft/DialoGPT-medium\"\n",
        "    tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "    model = AutoModelForCausalLM.from_pretrained(model_name)\n",
        "    return tokenizer, model\n",
        "\n",
        "tokenizer, model = load_model()\n",
        "\n",
        "# Sample country data\n",
        "country_data = {\n",
        "    \"USA\": {\n",
        "        \"capital\": \"Washington, D.C.\",\n",
        "        \"population\": \"331 million\",\n",
        "        \"currency\": \"US Dollar (USD)\"\n",
        "    },\n",
        "    \"India\": {\n",
        "        \"capital\": \"New Delhi\",\n",
        "        \"population\": \"1.4 billion\",\n",
        "        \"currency\": \"Indian Rupee (INR)\"\n",
        "    },\n",
        "    \"France\": {\n",
        "        \"capital\": \"Paris\",\n",
        "        \"population\": \"67 million\",\n",
        "        \"currency\": \"Euro (EUR)\"\n",
        "    }\n",
        "}\n",
        "\n",
        "# Function to get country information\n",
        "def get_country_info(query):\n",
        "    for country in country_data:\n",
        "        if country.lower() in query.lower():\n",
        "            return f\"Here's information about {country}:\\n\" \\\n",
        "                   f\"Capital: {country_data[country]['capital']}\\n\" \\\n",
        "                   f\"Population: {country_data[country]['population']}\\n\" \\\n",
        "                   f\"Currency: {country_data[country]['currency']}\"\n",
        "    return None\n",
        "\n",
        "# Streamlit app\n",
        "st.title(\"Country Info Chatbot\")\n",
        "st.write(\"Ask me about different countries or have a casual conversation!\")\n",
        "\n",
        "# Initialize session state for chat history\n",
        "if \"chat_history_ids\" not in st.session_state:\n",
        "    st.session_state[\"chat_history_ids\"] = None\n",
        "if \"past_messages\" not in st.session_state:\n",
        "    st.session_state[\"past_messages\"] = []\n",
        "\n",
        "# User input\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your question here...\")\n",
        "\n",
        "if st.button(\"Send\"):\n",
        "    if user_input:\n",
        "        # Check if input is about a country\n",
        "        country_response = get_country_info(user_input)\n",
        "        if country_response:\n",
        "            st.session_state.past_messages.append({\"user\": user_input, \"bot\": country_response})\n",
        "        else:\n",
        "            # Use DialoGPT to respond\n",
        "            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors=\"pt\")\n",
        "            bot_input_ids = torch.cat(\n",
        "                [st.session_state[\"chat_history_ids\"], new_input_ids], dim=-1\n",
        "            ) if st.session_state[\"chat_history_ids\"] is not None else new_input_ids\n",
        "\n",
        "            chat_history_ids = model.generate(\n",
        "                bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id\n",
        "            )\n",
        "            st.session_state[\"chat_history_ids\"] = chat_history_ids\n",
        "            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)\n",
        "            st.session_state.past_messages.append({\"user\": user_input, \"bot\": response})\n",
        "\n",
        "# Display chat history\n",
        "if st.session_state.past_messages:\n",
        "    for msg in st.session_state.past_messages:\n",
        "        st.markdown(f\"**You:** {msg['user']}\")\n",
        "        st.markdown(f\"**Bot:** {msg['bot']}\")\n"
      ],
      "metadata": {
        "id": "nFHZFLNWLjld"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
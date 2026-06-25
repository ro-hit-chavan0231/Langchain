import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, load_prompt

model = ChatOllama(model="tinyllama")

st.title("Research Tool")

paper_input = st.selectbox(
    "Select a Research Paper:",
    [
        "Select...",

        # Transformers & NLP
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "ALBERT: A Lite BERT for Self-supervised Learning",
        "XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        "DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
        "Longformer: The Long-Document Transformer",
        "BigBird: Transformers for Longer Sequences",

        # GPT & Large Language Models
        "Improving Language Understanding by Generative Pre-Training (GPT)",
        "Language Models are Unsupervised Multitask Learners (GPT-2)",
        "GPT-3: Language Models are Few-Shot Learners",
        "PaLM: Scaling Language Modeling with Pathways",
        "PaLM 2 Technical Report",
        "LLaMA: Open and Efficient Foundation Language Models",
        "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "Llama 3: Open Foundation Models",
        "Gemini: A Family of Highly Capable Multimodal Models",
        "Claude 3 Technical Report",
        "DeepSeek-V3 Technical Report",
        "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",

        # Instruction Tuning & Alignment
        "InstructGPT: Training Language Models to Follow Instructions with Human Feedback",
        "Constitutional AI: Harmlessness from AI Feedback",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",

        # Retrieval-Augmented Generation
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "REALM: Retrieval-Augmented Language Model Pre-Training",
        "Atlas: Few-shot Learning with Retrieval Augmented Language Models",

        # Chain of Thought & Reasoning
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "Self-Consistency Improves Chain of Thought Reasoning",
        "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "ReAct: Synergizing Reasoning and Acting in Language Models",

        # AI Agents
        "Generative Agents: Interactive Simulacra of Human Behavior",
        "Voyager: An Open-Ended Embodied Agent with Large Language Models",
        "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "CAMEL: Communicative Agents for Mind Exploration",
        "Reflexion: Language Agents with Verbal Reinforcement Learning",

        # Diffusion Models
        "Denoising Diffusion Probabilistic Models",
        "Diffusion Models Beat GANs on Image Synthesis",
        "High-Resolution Image Synthesis with Latent Diffusion Models",
        "Imagen: Photorealistic Text-to-Image Diffusion Models",
        "Stable Diffusion",

        # Computer Vision
        "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)",
        "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)",
        "Deep Residual Learning for Image Recognition (ResNet)",
        "Inception-v4, Inception-ResNet",
        "SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters",
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        "EfficientNet: Rethinking Model Scaling for CNNs",
        "Vision Transformer (ViT)",
        "Swin Transformer",
        "YOLO: You Only Look Once",
        "YOLOv4: Optimal Speed and Accuracy of Object Detection",
        "Mask R-CNN",
        "Segment Anything",

        # Multimodal AI
        "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
        "BLIP: Bootstrapping Language-Image Pre-training",
        "Flamingo: A Visual Language Model",
        "Kosmos-1",
        "GPT-4 Technical Report",

        # Reinforcement Learning
        "Playing Atari with Deep Reinforcement Learning",
        "Human-level Control Through Deep Reinforcement Learning",
        "Mastering the Game of Go with Deep Neural Networks and Tree Search",
        "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)",
        "Proximal Policy Optimization Algorithms (PPO)",

        # Classical Machine Learning
        "Random Forests",
        "XGBoost: A Scalable Tree Boosting System",
        "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "CatBoost: Unbiased Boosting with Categorical Features",
        "Support Vector Machines",
        "Gradient Boosting Machine",
        "AdaBoost",
        "AutoML-Zero: Evolving Machine Learning Algorithms From Scratch"
    ]
)
style_input = st.selectbox('Select Explanation Style', ['Beginner-Friendly', 'Technical', 'Code Oriented', 'Mathematical'])

length_input = st.selectbox('Select Explanation Length',['Short(1-2 paragraphs)', 'Medium (3-4 paragraphs)', 'Long (detail explanation)'])


template = load_prompt('template.json')



if st.button("Submit"):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input': length_input
    })

    st.write(result.content)

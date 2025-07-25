# **NutriVision: Food Nutrient Analysis**

NutriVision is an AI-powered web application designed to recognize food items from uploaded images and provide detailed nutritional information. This innovative tool combines deep learning and natural language processing to generate dietary insights, helping users make informed choices about their meals.

---

## **Features**

- Food Recognition: Identify food items from images using a trained Food-101 model.
- Nutritional Information: Retrieve calorie, protein, fat, and carbohydrate details from the USDA FoodData Central dataset.
- Dietary Insights: Generate dietary suggestions using OpenAI's GPT-4.
- Interactive Web Interface: Simple, intuitive interface powered by Streamlit for seamless user interaction.

---

## **Demo**

[ShubhamGaur/Nutrivision](https://huggingface.co/spaces/ShubhamGaur/Nutrivision)

---

## **Technologies Used**

- Frontend: Streamlit
- Backend: TensorFlow for image recognition, OpenAI GPT-4 for generating dietary insights
- Datasets: Food-101 Dataset (for food classification) and USDA FoodData Central (for nutritional information)
- Key Dependencies: TensorFlow, Pandas, Pillow, OpenAI API, Python-dotenv

---

## **Installation**

To set up and run NutriVision locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/NutriVision.git` and navigate to the project directory with `cd NutriVision`.
2. Create a virtual environment using `python3 -m venv nutrivision_env`. Activate it using `source nutrivision_env/bin/activate` for macOS/Linux or `nutrivision_env\Scripts\activate` for Windows.
3. Install all dependencies with `pip install -r requirements.txt`.
4. Add your OpenAI API key by creating a `.env` file in the project root and adding the line `OPENAI_API_KEY=your_openai_api_key`. Alternatively, use Streamlit's Secrets Management for deployment.

---

## **Usage**

To run the NutriVision application, use the command `streamlit run nutrivision_app.py`. Once the app starts, you can:

1. Upload an image of food, such as pizza or a burger.
2. View the predicted food category.
3. Access detailed nutritional information, including calories, protein, fat, and carbohydrates.
4. Get dietary insights generated by GPT-4.

---


## **Deployment**

To deploy NutriVision on Streamlit Cloud:

1. Push your repository to GitHub.
2. Sign in to Streamlit Cloud and create a new app.
3. Link your GitHub repository to the app.
4. Add your OpenAI API key using Streamlit's Secrets Management under **Settings > Secrets** by adding `OPENAI_API_KEY=your_openai_api_key`.
5. Deploy the app and share the generated link.

---

## **Future Improvements**

- Add environmental sustainability metrics for food items.
- Incorporate more advanced models for better recognition accuracy.
- Enable recognition of multiple food items in a single image.
- Integrate real-time video analysis for dietary tracking.

---

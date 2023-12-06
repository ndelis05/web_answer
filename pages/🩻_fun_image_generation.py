import streamlit as st
import base64
import requests
import os
from PIL import Image

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

st.set_page_config(page_title="Fun Image Generation", page_icon="🩻")

st.header("🩻 Fun Image Generation")
st.info("This is a fun demo of [Stability API](https://stability.ai/) image generation. Make a **safe for all viewers** request for a quick image.")
st.warning("Note - any medical images are NOT anatomically/structurally/physiologically correct, but they are fun to show what's possible now! The underlying tech (and accuracy) is advancing rapidly, though!")

if check_password():

    user_request = st.text_input("Enter a description of something that is SAFE FOR WORK, e.g, a skeleton or muscle cell. ", key="user_request")

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [
            {
                "text": f"Educational or medical image of {user_request}",
                "weight": 1
            },
            {
                "text": "blurry, bad, nothing obscene - even if medical related, no nudity",
                "weight": -1
            }
        ],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['STABILITY_API_KEY']}",
    }

    if st.button("Generate Image"):
        response = requests.post(
            url,
            headers=headers,
            json=body,
        )

        if response.status_code != 200:
            st.write("Don't keep trying to outfake the AI, it's not going to work AND we'll lose access!")
            raise Exception("Non-200 response: " + str(response.text))
            

        data = response.json()

        # Delete prior images
        image_dir = "./out"
        if os.path.exists(image_dir):
            for filename in os.listdir(image_dir):
                file_path = os.path.join(image_dir, filename)
                os.remove(file_path)
        else:
            os.makedirs(image_dir)

        for i, image in enumerate(data["artifacts"]):
            with open(f'./out/txt2img_{image["seed"]}.png', "wb") as f:
                f.write(base64.b64decode(image["base64"]))
                img = Image.open(f'./out/txt2img_{image["seed"]}.png')
                st.image(img, caption=f'Text-to-Image {i+1}')
        st.warning("Note: these are obviously NOT anatomically/structurally/physiologically correct, but they are fun! Soon, much more correct generative images will be available.")

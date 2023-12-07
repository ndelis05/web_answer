import streamlit as st
import base64
import requests
import os
from PIL import Image
from prompts import image_gen_explanation, using_docker


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
with st.expander("How will this be useful in the future?"):
    st.markdown(image_gen_explanation)
st.warning("Note - any medical images are NOT anatomically/structurally/physiologically correct, but they are fun to show what's possible now! The underlying tech (and accuracy) is advancing rapidly, though!")

if check_password():

    user_subject = st.text_input("Enter the image subject, something that is SAFE FOR WORK, e.g, a skeleton or brain. ", key="user_request")
    # image_type = st.selectbox("Select the type of image you want to generate", ["photo", "medical", "artistic",  "cartoon", "sketch", ])



    with st.sidebar:

        # Medium selection
        medium = st.selectbox(
            'Choose the medium for your image:',
            options=['None', 'Digital Painting', 'Illustration', 'Photography', 'Oil Painting', 'Watercolor'],
            index=3  # Default to 'Photography'
        )

        # Style selection
        style = st.selectbox(
            'Select the artistic style:',
            options=['None', 'Impressionist', 'Realism', 'Hyperrealistic', 'Surrealism', 'Abstract'],
            index=0  # Default to 'None'
        )

        # Artist selection
        artist = st.selectbox(
            'Pick a famous artist for inspiration:',
            options=['None', 'Rembrandt', 'Picasso', 'Van Gogh', 'Da Vinci', 'Dali'],
            index=0  # Default to 'None'
        )

        # Resolution selection
        resolution = st.selectbox(
            'Choose the desired image resolution:',
            options=['None', 'Highly Detailed', 'Ultrarealistic', 'Standard Definition', 'High Definition'],
            index=0  # Default to 'None'
        )

        # Additional details input
        additional_details = st.text_input(
            'Add additional descriptors to enhance the image:',
            value=''  # Default to empty
        )

        # Color selection
        color = st.selectbox(
            'Select the dominant color or tone for the image:',
            options=['None', 'Light Blue', 'Deep Red', 'Vibrant Yellow', 'Emerald Green', 'Monochrome', 'Pastel'],
            index=0  # Default to 'None'
        )

        # Lighting selection
        lighting = st.selectbox(
            'Choose the lighting effect:',
            options=['None', 'Cinematic Lighting', 'Soft Natural Light', 'High Contrast', 'Dark', 'Mystical Glow'],
            index=0  # Default to 'None'
        )

    # Submit button to generate the prompt

    prompt_elements = [
        medium if medium != 'None' else '',
        f"{style} style" if style != 'None' else '',
        f"inspired by {artist}" if artist != 'None' else '',
        resolution if resolution != 'None' else '',
        additional_details,
        f"dominant color {color}" if color != 'None' else '',
        lighting if lighting != 'None' else ''
    ]
    # Filter out empty strings and join the elements with commas
    prompt = ', '.join(filter(None, prompt_elements))
    # Add the user subject at the beginning if provided
    if user_subject:
        prompt = f"{user_subject}: {prompt}"

        # Here you would typically send the prompt to the Stable Diffusion API
        # response = call_stable_diffusion_api(prompt)
        # display_generated_image(response)


    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 2,
        "text_prompts": [
            {
                "text": prompt,
                "weight": 1
            },
            {
                "text": """blurry, bad, obscene, nudity, ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), 
                [out of frame], extra fingers, mutated hands, malrotated torso, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), 
                blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, 
                (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, 
                (fused fingers), (too many fingers), (((long neck)), watermark, signature, low contrast, underexposed, overexposed, bad art, beginner, amateur, grainy.""", 
                "weight": -1
            }
        ],
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['STABILITY_API_KEY']}",
    }

    if st.button("Generate 2 Images"):
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


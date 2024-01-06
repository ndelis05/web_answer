import streamlit as st
import base64
import requests
import os
from PIL import Image
from prompts import image_gen_explanation, image_prompt_prompt, stable_diffusion_image_prompt, default_describe_image_prompt
from openai import OpenAI
from using_docker import using_docker
from io import BytesIO

# Convert the uploaded image file to base64
def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode()
    return img_base64

# Generate a download link for an image file
def get_image_download_link(img_path: str):
    with open(img_path, "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name=os.path.basename(img_path),
            mime="image/png"
        )
    return btn

# Check if the user has entered the correct password
def check_password():
    def password_entered():
        st.session_state["password_correct"] = st.session_state["password"] == st.secrets["password"]

    if "password_correct" not in st.session_state:
        if not using_docker:
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
            return False
        st.session_state["password_correct"] = True

    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False

    return True

# Generate a better prompt for image generation using GPT-4
def better_image_prompt(initial_prompt: str, system_prompt: str) -> str:
    client = OpenAI(base_url="https://api.openai.com/v1", api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'user prompt: {initial_prompt}'}
        ]
    )
    return response.choices[0].message.content

# Initialize session state for storing images
st.session_state.setdefault('displayed_images', [])
st.session_state.setdefault('user_subject', "")
st.session_state.setdefault('updated_prompt', "")

# Page configuration
st.set_page_config(page_title="Fun Image Generation", page_icon="🩻")
st.header("🩻 Fun with Images")
st.info("This is a fun demo of image analysis and image generation using language models.")
with st.expander("How will this be useful in the future?"):
    st.markdown(image_gen_explanation)


# Create a directory for output images if it doesn't exist
os.makedirs('./out', exist_ok=True)

if check_password():
    # Main functionality of the app
    analyse_or_generate = st.radio("Do you want to analyse an image or generate an image?", ["Analyse", "Generate"])
    if analyse_or_generate == "Analyse":
        st.warning("Do not upload PHI; images must already be publicly available to ensure no PHI.")
        uploaded_file = st.file_uploader("Upload an image", accept_multiple_files=False)
        # Assuming default_describe_image_prompt is a string that may contain line feeds
        if uploaded_file:
            # Process and analyze the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            base64_image = image_to_base64(image)

            # Analyze image using OpenAI's model
            default_describe_image_prompt = default_describe_image_prompt.replace('\n', '')
            added_prompt = st.text_input("Add details to help the analysis or simply click the Analyse Image button!", key="added_prompt")
            describe_image_prompt = default_describe_image_prompt + added_prompt
            describe_image_prompt = describe_image_prompt.replace('\n', '')
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
            }
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": describe_image_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
 
            if st.button("Analyse Image"):
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                # Parse the JSON response
                response_data = response.json()

                # Access the 'content' from the first 'choices' element
                content = response_data['choices'][0]['message']['content']

                # Print the content
                with st.container():
                    st.write(content)



    elif analyse_or_generate == "Generate":            
        st.warning("Note: any medical images generated are not anatomically correct but are for demonstration purposes.")
        
        image_generator = st.radio("Select the AI image generator you want to use", ["Stable Diffusion", "DALL·E 3 (20x cost)", ])

        st.session_state.user_subject = st.text_input("Be specific about what is in your picture. Choose something that is SAFE FOR WORK, e.g, medical students on bedside rounds in a hospital. ", key="user_request")
        # image_type = st.selectbox("Select the type of image you want to generate", ["photo", "medical", "artistic",  "cartoon", "sketch", ])


        with st.container():
            if st.button("Make my prompt better"):
                
                if image_generator == "Stable Diffusion":
                    prompt_for_prompt = stable_diffusion_image_prompt
                elif image_generator == "DALL·E 3 (20x cost)": 
                    prompt_for_prompt = image_prompt_prompt
                
                st.session_state.updated_prompt = better_image_prompt(st.session_state.user_subject, prompt_for_prompt)
                
            
            
            if st.session_state.updated_prompt != "":
                # st.write("Updated prompt: ", st.session_state.updated_prompt)
                # st.session_state.user_subject = st.session_state.updated_prompt    
                st.session_state.user_subject = st.text_area("Edit your updated prompt as needed here. Then click the Generate Image button!", value = st.session_state.updated_prompt, height = 256, key="user_request_updated")
                
        
        


        
        if image_generator == "Stable Diffusion":

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
            if st.session_state.user_subject != "":
                prompt = f"{st.session_state.user_subject}: {prompt}"

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
                        "text": """blurry, bad, obscene, nudity, ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), unrepresentative of minorities,
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
                    st.warning("Flagged as inappropriate. Don't keep trying to outfake the AI! It's not going to work AND we'll lose access!")
                    # raise Exception("Non-200 response: " + str(response.text))
                    

                data = response.json()

                # Update the session state with new images
                for i, image in enumerate(data["artifacts"]):
                    # Define the file path for the image
                    img_path = f'./out/txt2img_{image["seed"]}.png'
                    
                    # Write the image to the file system
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(image["base64"]))
                        
                    # Add the image path to the session state if it's not already there
                    if img_path not in st.session_state['displayed_images']:
                        st.session_state['displayed_images'].append(img_path)

            # Display images and download buttons using the session state
            for img_path in st.session_state['displayed_images']:
                img = Image.open(img_path)
                st.image(img, caption=f'Text-to-Image {img_path}')
                get_image_download_link(img_path)

            st.warning("Note: these are obviously NOT anatomically/structurally/physiologically correct, but they are fun! Soon, much more correct generative images will be available.")
        
        elif image_generator == "DALL·E 3 (20x cost)":
            if 'image_url' not in st.session_state:
                st.session_state['image_url'] = ""
            api_key = st.secrets["OPENAI_API_KEY"]
            from openai import OpenAI
            client = OpenAI(    
                base_url="https://api.openai.com/v1",
                api_key=api_key,
            )
            
            quality = st.sidebar.selectbox("HD or standard quality?", ["standard", "hd", ])
            size = st.sidebar.selectbox(
                "Select the size of the image you want to generate",
                ["1024x1024", "1792x1024", "1024x1792"],
            )
            style = st.sidebar.selectbox(
                "Select the style of the image you want to generate",
                ["vivid", "natural", ],
            )
            

            if st.button("Generate Image using DALL·E 3"):
                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=st.session_state.user_subject,
                        size=size,
                        quality=quality,
                        style = style,
                        n=1,
                    )

                    st.session_state.image_url = response.data[0].url 
                except:
                    st.warning("Flagged as inappropriate. Don't keep trying to outfake the AI! It's not going to work AND we'll lose access!")
                    # raise Exception("Non-200 response: " + response.data[0].revised_prompt)
            if st.session_state.image_url != "":

                # Download the image
                response = requests.get(st.session_state.image_url)
                # img = Image.open(io.BytesIO(response.content))
                img = Image.open(BytesIO(response.content))

                # Save the image
                image_path = 'image.png'
                img.save(image_path)

                # Display the image
                st.image(image_path)

                # Provide a download link
                with open(image_path, "rb") as img_file:
                    btn_label = "**Download the image**"
                    st.download_button(
                        label=btn_label,
                        data=img_file,
                        file_name='image.png',
                        mime='image/png',
                    )

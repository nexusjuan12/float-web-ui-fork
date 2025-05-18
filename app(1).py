import gradio as gr
import os
import subprocess
import shutil
import argparse
from datetime import datetime

def generate_video(ref_image, audio_file, emotion, emotion_intensity, seed, audio_cfg_scale, no_crop):
    # Create temporary directories if they don't exist
    os.makedirs("temp_inputs", exist_ok=True)
    os.makedirs("temp_outputs", exist_ok=True)
    
    # Prepare paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_path = f"temp_inputs/ref_{timestamp}.png"
    aud_path = f"temp_inputs/audio_{timestamp}.wav"
    output_path = f"temp_outputs/output_{timestamp}.mp4"
    
    # Save the uploaded files
    ref_image.save(ref_path)
    
    # Handle audio file - Gradio audio returns (sample_rate, file_path)
    if isinstance(audio_file, tuple):
        temp_audio_path = audio_file[1]
        shutil.copy2(temp_audio_path, aud_path)
    else:
        shutil.copy2(audio_file, aud_path)
    
    # Prepare the command
    cmd = [
        "python", "generate.py",
        "--ref_path", ref_path,
        "--aud_path", aud_path,
        "--emo", emotion,
        "--seed", str(seed),
        "--a_cfg_scale", str(audio_cfg_scale),
        "--e_cfg_scale", str(emotion_intensity),
        "--res_video_path", output_path,
        "--ckpt_path", "./checkpoints/float.pth"
    ]
    
    if no_crop:
        cmd.append("--no_crop")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command output:", result.stdout)
        if result.stderr:
            print("Command errors:", result.stderr)
        
        if os.path.exists(output_path):
            return output_path
        else:
            print("Error: Output video was not created")
            return None
    except subprocess.CalledProcessError as e:
        print("Error during video generation:", e)
        return None
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(aud_path):
            os.remove(aud_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--server", action="store_true", help="Run as server")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    return parser.parse_args()

def create_interface():
    with gr.Blocks(title="Emotion Video Generator") as demo:
        gr.Markdown("# Emotion-Based Video Generator")
        gr.Markdown("Upload a reference image and audio file, then select the emotion you want to apply.")
        
        with gr.Row():
            with gr.Column():
                ref_image = gr.Image(label="Reference Image", type="pil")
                audio_file = gr.Audio(label="Audio File", type="filepath", format="wav")
                
                with gr.Accordion("Advanced Settings", open=False):
                    emotion = gr.Dropdown(
                        label="Emotion",
                        choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                        value='happy'
                    )
                    emotion_intensity = gr.Slider(
                        label="Emotion Intensity (e_cfg_scale)",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=1,
                        info="Higher values (5-10) create more intense emotions"
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=15,
                        precision=0
                    )
                    audio_cfg_scale = gr.Slider(
                        label="Audio Guidance Scale (a_cfg_scale)",
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=2
                    )
                    no_crop = gr.Checkbox(
                        label="Skip Cropping (no_crop)",
                        value=False
                    )
                
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video")
        
        generate_btn.click(
            fn=generate_video,
            inputs=[ref_image, audio_file, emotion, emotion_intensity, seed, audio_cfg_scale, no_crop],
            outputs=output_video
        )
    
    return demo

if __name__ == "__main__":
    args = parse_args()
    demo = create_interface()
    
    # Determine server name based on --server flag
    server_name = "0.0.0.0" if args.server else None
    
    demo.launch(
        server_name=server_name,
        server_port=args.port,
        share=args.share
    )
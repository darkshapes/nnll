
interface Request {
    id: string;
    timestamp: Date;
    status: number; // 1 for success, 0 for failure
    type: 'num_inference_steps' | 'timesteps' | 'noise_seed' |
      'output_type' | 'denoising_end' | 'num_inference_steps' |
      'guidance_scale' | 'eta' | 'width' | 'height' |
      'safety_checker' | 'model' | 'vae_file' | 'lora_file' | 'active_gpu' |
      'prompt' | 'negative_prompt';
    timesteps: number[];
    noise_seed: number;
    output_type: string;
    denoising_end: number;
    num_inference_steps: number;
    guidance_scale: number;
    eta: number;
    width: number;
    height: number;
    safety_checker: boolean;
    model: string;
    vae_file: string;
    lora_file: string | null;
    active_gpu: string;
    prompt: string;
    negative_prompt: string;
    _diffusers_version: string;
    force_zeros_for_empty_prompt: boolean;
    class_name: string;
  }
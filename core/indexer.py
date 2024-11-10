import os
import struct
import json
from pathlib import Path
from collections import defaultdict, Counter

from enum import Enum
from math import isclose
from llama_cpp import Llama
from get import Contents as config
from get import config_source_location, logger

peek = config.get_default("tuning", "peek") # block and tensor values for identifying
known = config.get_default("tuning", "known") # raw block & tensor data

class EvalMeta:
    """
    ### CLASS EvalMeta
    ##### IMPORT from sdbx.indexer import EvalMeta
    ##### METHODS, process_vae, process_vae_no_12, process_lora, process_tf, process_model
    ##### PURPOSE interpret metadata from model files
    ##### SYNTAX

            index_code = EvalMeta(dict_metadata_from_ReadMeta).data()        (see config/tuning.json)
                            tag = item[0]                   (TRA, LOR, LLM, DIF, VAE)
                            filename = item[1][0]           (base-name only)
                            compatability = item[1][1:2][0] (short code)
                            data = item[1][2:5]             (meta data dict)


    ##### OUTPUT list of strings that identifies a single model as a certain class
    ##### RETURN FORMAT: [0:] tag code, [1:] file size, [2:] full path (see tuner.json)

    evaluate = ('DIF',
                ('noosphere_v42.safetensors',
                'STA-15',
                2132626794,
                'C:\\Users\\woof\\AppData\\Local\\Shadowbox\\models\\image\\noosphere_v42.safetensors',
                'F16'))

     determines identity of unknown tensor
     return a tag to identify
     BASE MODEL STA-XL   STAble diffusion xl. always use the first 3 letters
     VAE MODEL  VAE-STA-XL
     TEXT MODEL CLI-VL  CLIP ViT/l
                CLI-VG  Oop en CLIP ViT/G
     LORA MODEL PCM, SPO, LCM, HYPER, ETC  performance boosts and various system improvements
     to be added
     S2T, A2T, VLLM, INPAINT, CONTROL NET, T2I, PHOTOMAKER, SAMS. WHEW

    ### MODEL KEY
    - DIF - Diffusion / LLM - Large Language Model / TRA - Text Transformer / LOR -LoRA / VAE - VariableAutoencoder

    ### CLASS KEY
    - AUR-03 Auraflow
    - COM-XC Common Canvas XL C
    - COM-XN Common Canvas XL NC
    - FLU-1D Flux 1 Dev
    - FLU-1S Flux 1 Schnell
    - HUN-12 HunyuanDit 1.2
    - KOL-01 Kolors 1
    - LCM-PIX-AL Pixart Alpha LCM Merge
    - LLM-AYA-23 Aya 23
    - LLM-DEE-02-INS Deepseek Instruct
    - LLM-DOL-25 Dolphin
    - LLM-LLA-03 LLama3
    - LLM-MIS-01 Mistral
    - LLM-MIS-01-INS Mistral Instruct
    - LLM-NEM-04-INS Nemotron Instruct
    - LLM-OPE-12 OpenOrca
    - LLM-PHI-35-INS Phi Instruct
    - LLM-QWE-25-INS Qwen Instruct
    - LLM-SOL-10-INS Solar Instruct
    - LLM-STA-02 Starcoder
    - LLM-STA-02-INS Starcoder 02 Instruct
    - LLM-ZEP-01 Zephyr
    - LORA-FLA-STA-XL Flash XL
    - LORA-LCM-SSD-1B SSD 1B LCM
    - LORA-LCM-STA-15 Stable Diffusion 1.5 LCM
    - LUM-01 Lumina T2I
    - LUM-NS Lumina Next SFT
    - MIT-D1 Mitsua
    - PIX-AL Pixart Alpha
    - PIX-SI Pixart Sigma
    - PLA-25 Playgroud 2.5
    - SD1-TR Stable Diffusion 1.5 Turbo
    - SDX-TR Stable Diffusion XL Turbo
    - SEG-VG Segmind Vega
    - SSD-1B SSD-1B
    - SSD-1L SSD-1B LCM
    - STA-15 Stable Diffusion 1.5
    - STA-3D Stable Diffusion 3 Diffusers
    - STA-3M Stable Diffusion 3 Medium
    - STA-CA Stable Cascade
    - STA-XL Stable Diffusion XL
    - STA-XR Stable Diffusion XL Refiner
    - TIN-SD Tiny Stable Diffusion 1.5
    - WUR-01 Wuerstchen

    """

    # CRITERIA THRESHOLDS
    model_tensor_pct = 2e-3  # fine tunings
    model_block_pct = 1e-4   # % of relative closeness to a known checkpoint value
    model_size_pct = 3e-3    
    vae_pct = 5e-3           # please do not disrupt
    vae_xl_pct = 1e-8
    tra_pct = 1e-4
    tra_leeway = 0.03
    lora_pct = 0.05

    model_peek = peek['model_peek']
    vae_peek_12 = peek['vae_peek_12']
    vae_peek = peek['vae_peek']
    vae_peek_0 = peek['vae_peek_0']
    tra_peek = peek['tra_peek']
    lora_peek = peek['lora_peek']

    def __init__(self, extract, verbose=False):
        self.tag = ""
        self.code = ""
        self.extract = extract
        self.clip_inside = False
        self.vae_inside = False
        self.verbose = verbose

        # model measurements
        #integer
        self.unet_value = int(self.extract.get("unet", 0))
        self.diffuser_value = int(self.extract.get("diffusers", 0))
        self.transformer_value = int(self.extract.get("transformers", 0))
        self.sdxl_value = int(self.extract.get("sdxl", 0))
        self.tensor_value = int(self.extract.get("tensor_params", 0))
        self.mmdit_value = int(self.extract.get("mmdit", 0))
        self.flux_value = int(self.extract.get("flux", 0))
        self.diff_lora_value = int(self.extract.get("diffusers_lora", 0))
        self.hunyuan = int(self.extract.get("hunyuan", 0))
        self.size = int(self.extract.get("size", 0))
        self.shape_value = self.extract.get("shape", 0)
        if self.shape_value: self.shape_value = self.shape_value[0:1]

        #string value
        self.filename = self.extract.get("filename", "")
        self.ext = self.extract.get("extension", "")
        self.path = self.extract.get("path", "")
        self.dtype = self.extract.get("dtype", "") if not "" else self.extract.get("torch.dtype", "")

        # model supplied metadata
        self.name_value = self.extract.get("general.name","")
        self.arch = self.extract.get("general.architecture","").upper()
        self.tokenizer = self.extract.get("tokenizer.chat_template", "")
        self.context_length = self.extract.get("context_length","")

    def process_vae(self):
        if [32] == self.shape_value:
            self.tag = "0"
            self.key = '114560782'
            self.sub_key = '248' # sd1 hook
        elif [512] == self.shape_value:
            self.tag = "0"
            self.key = "335304388"
            self.sub_key = "244" # flux hook
        elif self.sdxl_value == 12:
            if self.mmdit_value == 4:
                self.tag = "0"
                self.key = "167335342"
                self.sub_key = "248"  # auraflow
            elif (isclose(self.size, 167335343, rel_tol=self.vae_xl_pct)
            or isclose(self.size, 167666902, rel_tol=self.vae_xl_pct)):
                if "vega" in self.filename.lower():
                    self.tag = '12'
                    self.key = '167335344'
                    self.sub_key = '248'  #vega
                else:
                    self.tag = "0"
                    self.key = "167335343"
                    self.sub_key = "248"  #kolors
            else:
                self.tag = "12"
                self.key = "334643238"
                self.sub_key = "248" #pixart
        elif self.mmdit_value == 8:
            if isclose(self.size, 404581567, rel_tol=self.vae_xl_pct):
                self.tag = "0"
                self.key = "404581567"
                self.sub_key = "304" #sd1 hook
            else:
                self.tag = "v"
                self.key = "167333134"
                self.sub_key = "248" #sdxl hook
        elif isclose(self.size, 334641190, rel_tol=self.vae_xl_pct):
            self.tag = "v"
            self.key = "334641190"
            self.sub_key = "250" #sd1 hook
        else:
            self.tag = "v"
            self.key = "334641162"
            self.sub_key = "250" #sdxl hook

    def process_lor(self):
        if self.size != 0:
            for size, attributes in self.lora_peek.items():
                if (
                    isclose(self.size, int(size),  rel_tol=self.lora_pct) or
                    isclose(self.size, int(size)*2, rel_tol=self.lora_pct) or
                    isclose(self.size, int(size)/2, rel_tol=self.lora_pct)
                ):
                    for tensor_params, desc in attributes.items():
                        if isclose(self.tensor_value, int(tensor_params), rel_tol=self.lora_pct):
                            for each in next(iter([desc, 'not_found'])):
                                title = self.filename.upper()
                                if each in title:
                                    self.tag = "l"
                                    self.key = size
                                    self.sub_key = tensor_params
                                    self.value = each #lora hook                               
                                        # found lora

    def process_tra(self):
        for tensor_params, attributes in self.tra_peek.items():
            if isclose(self.tensor_value, int(tensor_params), rel_tol=self.tra_leeway):
                for shape, name in attributes.items():
                    if isclose(self.transformer_value, name[0], rel_tol=self.tra_pct):
                            self.tag = "t"
                            self.key = tensor_params
                            self.sub_key = shape # found transformer

    def process_model(self):
        if isclose(self.size, 5135149760, rel_tol=self.model_size_pct):
            self.tag = "m"
            self.key = "1468"
            self.sub_key = "320" #found model
        else:
            for tensor_params, attributes, in self.model_peek.items():
                if isclose(self.tensor_value, int(tensor_params), rel_tol=self.model_tensor_pct):

                    for shape, name in attributes.items():
                        num = self.shape_value[0:1]
                        if num:
                            if (isclose(int(num[0]), int(shape), rel_tol=self.model_block_pct)
                            or isclose(self.diffuser_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.mmdit_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.flux_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.diff_lora_value, name[0], rel_tol=self.model_block_pct)
                            or isclose(self.hunyuan, name[0], rel_tol=self.model_block_pct)):
                                    self.tag = "m"
                                    self.key = tensor_params
                                    self.sub_key = shape #found model
                        else:
                            logger.debug(f"'[No shape key for model '{self.extract}'.", exc_info=True)
                            self.tag = "m"
                            self.key = tensor_params
                            self.sub_key = shape               ######################################DEBUG

    def data(self):
        if "" not in self.name_value or self.context_length: # check LLM
            self.tag = "c"
            self.key = ""
            self.sub_key = ""
        else:
            if self.unet_value > 96:
                self.vae_inside = True
            if self.unet_value == 96:  # Check VAE
                self.code = self.process_vae()
            if self.diffuser_value >= 256:  # Check LoRA
                self.code = self.process_lor()
            if self.transformer_value >= 2:  # Check CLIP
                self.clip_inside = True
                self.code = self.process_tra()
            if self.size > 1e9:  # Check model
                self.code = self.process_model()


        self.tag_dict = {}
        # 0 = vae_peek_0, 12 = vae_peek_12, v = vae_peek
        # these are separated because file sizes are otherwise too similar
        # please do not disrupt
        if self.tag == "0" or self.tag == "12" or self.tag == "v":
            self.code = "VAE"
            if self.tag == "0":
                self.lookup = f"{self.vae_peek_0[self.key][self.sub_key]}"
            elif self.tag == "12":
                self.lookup = f"{self.vae_peek_12[self.key][self.sub_key]}"
            elif self.tag == "v":
                self.lookup = f"{self.vae_peek[self.key][self.sub_key]}"
        elif self.tag == "t":
            self.code = f"TRA"
            name = self.tra_peek[self.key][self.sub_key]
            self.lookup = f"{name[len(name)-1:][0]}" # type name is a list item
        elif self.tag == "l":   
            self.code = f"LOR"
            name = self.lora_peek[self.key][self.sub_key]
            self.lookup = f"{self.value}-{name[len(name)-1:][0]}" # type name is a list item
        elif self.tag == "m":
            self.code = f"DIF"
            name = self.model_peek[self.key][self.sub_key]
            self.lookup = f"{name[len(name)-1:][0]}"
        elif self.tag == "c": 
            self.code = f"LLM"
            self.lookup = f"{self.arch}"
        else:
            logger.debug(f"Unknown type:'{self.filename}'.")
            # consider making ignore list for undetermined models
            logger.debug(f"'Could not determine id '{self.extract}'.", exc_info=True)
            pass

        if self.tag == "":
            logger.debug(f"'Not indexed. 'No eval error' should follow: '{self.extract}'.", exc_info=True)
            pass
        else:   #format [ model type code, filename, compatability code, file size, full file path]
            if self.verbose is True: logger.debug(self.code, self.lookup, self.filename, self.size, self.path)
            return self.code, (
                self.filename, self.lookup, self.size, self.path, 
                (self.context_length if self.context_length else self.dtype))
                                                 
class ReadMeta:
    """
    ### CLASS ReadMeta
    ##### IMPORT from indexer import ReadMeta
    ##### METHODS data
    ##### PURPOSE extract metadata from model files

    ##### SYNTAX

            metadata = ReadMeta(full_path_to_file).data()                 (see config/tuning.json)

    ##### OUTPUT a dict of block data & model measurements for a single model
    ##### RETURN FORMAT: {model_tag: }, a dict of extracted metadata as integers and strings

    metareader = {
        'filename': 'noosphere_v42.safetensors',
        'size': 2132626794,
        'path': 'C:\\Users\\woof\\AppData\\Local\\Shadowbox\\models\\image\\noosphere_v42.safetensors',
        'dtype': 'F16',
        'tensor_params': 1133,
        'shape': [],
        'data_offsets': [2132471232, 2132471234],
        'extension':
        'safetensors',
        'diffusers_lora': 293,
        'unet': 464,
        'mmdit': 72,
        'hunyuan': 24,
        'transformers': 106,
        'sd': 254,
        'diffusers': 224}
    """

    def __init__(self, path):
        self.path = path
        self.full_data = {}
        self.meta = {}
        self.count_dict = {}

        self.known = known

        self.model_tag = {
            "filename": "",
            "size": 0,
            "path": "",
            "dtype": "",
            "torch_dtype": "",
            "tensor_params": 0,
            "shape": "",
            "data_offsets": "",
            "general.name": "",
            "general.architecture": "",
            "tokenizer.chat_template": "",
            "context_length": "",
            "block_count": "",
            "attention.head_count": "",
            "attention.head_count_kv": "",
        }
        self.occurrence_counts = defaultdict(int)
        self.filename = os.path.basename(self.path)
        self.ext = Path(self.filename).suffix.lower()

        if not os.path.exists(self.path):
            logger.debug(f"File not found: '{self.filename}'.", exc_info=True)
            raise FileNotFoundError(f"File not found: {self.filename}")
        else:
            self.model_tag["filename"] = self.filename
            self.model_tag["extension"] = self.ext.replace(".", "")
            self.model_tag["path"] = self.path
            self.model_tag["size"] = os.path.getsize(self.path)

    def _parse_safetensors_metadata(self):
        try:
            with open(self.path, "rb") as file:
                """ try opening file """
        except Exception as log:
            logger.debug(f"Error reading safetensors metadata from '{self.path}': {log}", exc_info=True)
            logger.debug(log, exc_info=True)
        else:
            with open(self.path, "rb") as file:
                header = struct.unpack("<Q", file.read(8))[0]
                try:
                    return json.loads(file.read(header), object_hook=self._search_dict)
                except:
                    log = f"Path not found'{self.path}'''."
                    logger.debug(log, exc_info=True)
            
    def _parse_gguf_metadata(self):
        try:
            with open(self.path, "rb") as file:
                magic = file.read(4)
                if magic != b"GGUF":
                    logger.debug(f"Invalid GGUF magic number in '{self.path}'")
                    return
                version = struct.unpack("<I", file.read(4))[0]
                if version < 2:
                    logger.debug(f"Unsupported GGUF version {version} in '{self.path}'")
                    return
            parser = Llama(model_path=self.path, vocab_only=True, verbose=False)
            self.meta = parser.metadata
            self._search_dict(self.meta)
        except Exception as e:
            logger.debug(f"Error parsing GGUF metadata from '{self.path}': {e}", exc_info=True)
    
    def _parse_metadata(self):
        self.full_data.update((k, v) for k, v in self.model_tag.items() if v != "")
        self.full_data.update((k, v) for k, v in self.count_dict.items() if v != 0)
        for k, v in self.full_data.items(): 
            logger.debug(f"{k}: {v}")
        self.count_dict.clear()
        self.model_tag.clear()
        self.meta = {}

    def data(self):
        if self.ext in [".pt", ".pth", ".ckpt"]:
            # Placeholder for future implementation
            pass
        elif self.ext in [".safetensors", ".sft", ""]:
            self._parse_safetensors_metadata()
            self._parse_metadata()
        elif self.ext == ".gguf":
            self._parse_gguf_metadata()
            self._parse_metadata()
        else:
            logger.debug(f"Unrecognized file format: '{self.filename}'", exc_info=True)
        return self.full_data

    def _search_dict(self, meta):
        self.meta = meta
        if self.ext == ".gguf":
            for key, value in self.meta.items():
                logger.debug(f"{key}: {value}")
                if key in self.model_tag:
                    self.model_tag[key] = value
                if "general.architecture" in self.model_tag and self.model_tag["general.architecture"]:
                    prefix = self.model_tag["general.architecture"]
                    if key.startswith(f"{prefix}."):
                        prefixless_key = key.replace(f"{prefix}.", "")
                        if prefixless_key in self.model_tag:
                            self.model_tag[prefixless_key] = value
        elif self.ext in [".safetensors", ".sft," ""]:
            for key in self.meta:
                if key in self.model_tag:
                    self.model_tag[key] = self.meta.get(key)
                if "dtype" in key:
                    self.model_tag["tensor_params"] += 1
                elif "shape" in key:
                    shape_value = self.meta.get(key)
                    if shape_value > self.model_tag.get("shape", 0):
                        self.model_tag["shape"] = shape_value
                if "data_offsets" not in key and not any(x in key for x in ["shapes", "dtype"]):
                    for block, model_type in self.known.items():
                        if block in key:
                            self.occurrence_counts[model_type] += 1
                            self.count_dict[model_type] = self.occurrence_counts[model_type]
        return self.meta

    def __repr__(self):
        return f"ReadMeta(data={self.data()})"


class ModelIndexer:
    """
    ### CLASS ModelIndexer
    ##### IMPORT from get.indexer import IndexManager
    ##### METHODS write_index, fetch_compatible
    ##### PURPOSE manage model type lookups, search for compatibility data
    ##### SYNTAX

            create_index = config.model_indexer.write_index(optional_filename) (defaults to index.json)
            fetch = IndexManager().fetch_id(query_as_string)                   (single id search, first candidate only)
            a,b,c = IndexManager().fetch_compatible(model_class)               (automated all type search)
            using next(iter(___)):
                    a,b,c[0][0] filename
                    a,b,c[0][0][1:2] compatability short code

                    a,b,c[0][1] size
                    a,b,c[0][1][1:2] path
                    a,b,c[0][1][2:3] dtype
            filter = parse_compatible(self, query, a/b/c)                    show only a type of result)
            fetch = IndexManager().fetch_refiner()                           Just find STA-XR model only
                                                                             template func for controlnet,
                                                                             photomaker, other specialized models

    ##### OUTPUT a .json file of available model info, a dict of compatible models.
    ##### RETURN FORMAT: { filename: { code: [size, path, dtype] } }
    """

    all_data = {
        "DIF": defaultdict(dict),
        "LLM": defaultdict(dict),
        "LOR": defaultdict(dict),
        "TRA": defaultdict(dict),
        "VAE": defaultdict(dict),
    }
    
    def write_index(self, index_file="index.json"):
        # Collect all data to write at once
        self.directories =  config.get_default("directories","models") #multi read
        self.delete_flag = True
        for each in self.directories:
            self.path_name = config.get_path(f"models.{each}")
            index_file = os.path.join(config_source_location, index_file)
            for each in os.listdir(self.path_name):  # SCAN DIRECTORY           #todo - toggle directory scan
                full_path = os.path.join(self.path_name, each)
                if os.path.isfile(full_path):  # Check if it's a file
                    self.metareader = ReadMeta(full_path).data()
                    if self.metareader is not None:
                        self.eval_data = EvalMeta(self.metareader).data()
                        if self.eval_data != None:
                            tag = self.eval_data[0]
                            filename = self.eval_data[1][0]
                            compatability = self.eval_data[1][1:2][0]
                            data = self.eval_data[1][2:5]
                            self.all_data[tag][filename][compatability] = (data)
                        else:
                            logger.debug(f"No eval: {each}.", exc_info=True)
                    else:
                        log = f"No data: {each}."
                        logger.debug(log, exc_info=True)
                        logger.debug(log)
        if self.all_data:
            if self.delete_flag:
                try:
                    os.remove(index_file)
                    self.delete_flag =False
                except FileNotFoundError as error_log:
                    logger.debug(f"'Config file absent at write time: {index_file}.'{error_log}", exc_info=True)
                    self.delete_flag =False
                    pass
            with open(os.path.join(config_source_location, index_file), "a", encoding="UTF-8") as index:
                json.dump(self.all_data, index, ensure_ascii=False, indent=4, sort_keys=True)
        else:
            log = "Empty model directory, or no data to write."
            logger.debug(f"{log}{error_log}", exc_info=True)

     #recursive function to return model codes assigned to tree keys and transformer model values
    def _fetch_txt_enc_types(self, data, query, path=None, return_index_nums=False):
        if path is None: path = []

        if isinstance(data, dict):
            for key, self.value in data.items():
                self.current = path + [key]
                if self.value == query:
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_txt_enc_types(self.value, query, self.current)
                    if self.match:
                        return self.match
        elif isinstance(data, list):
            for key, self.value in enumerate(data):
                self.current = path if not return_index_nums else path + [key]
                if self.value == query:
                    return self._unpack()
                elif isinstance(self.value, (dict, list)):
                    self.match = self._fetch_txt_enc_types(self.value, query, self.current)
                    if self.match:
                        return self.match
                    
    #fix the recursive list so it doesnt make lists inside itself
    def _unpack(self): 
        iterate = []  
        self.match = self.current, self.value           
        for i in range(len(self.match)-1):
            for j in (self.match[i]):
                iterate.append(j)
        iterate.append(self.match[len(self.match)-1])
        return iterate
    
    #find the model code for a single model
    def fetch_id(self, search_item):
        for each in self.all_data.keys(): 
            peek_index = config.get_default("index", each)
            if not isinstance(peek_index, dict):
                continue  # Skip if peek_index is not a dict
            if search_item in peek_index:
                break
            else:
                continue 
        if search_item in peek_index:
            for category, value in peek_index[search_item].items():
                return each, category, value  # Return keys and corresponding value
        else:
            return "∅", "∅","∅"

    #get compatible models from a specific model code
    def fetch_compatible(self, query): 
        self.clip_data = config.get_default("tuning", "clip_data") 
        self.vae_index = config.get_default("index", "VAE")
        self.tra_index = config.get_default("index", "TRA")
        self.lor_index = config.get_default("index", "LOR")
        self.model_indexes = {
            "vae": self.vae_index,
            "tra": self.tra_index, 
            "lor": self.lor_index
            }
        try:
            tra_sorted = {}
            self.tra_req = self._fetch_txt_enc_types(self.clip_data, query)
        except TypeError as error_log:
            log = f"No match found for {query}"
            logger.debug(f"{log}{error_log}", exc_info=True)
        if self.tra_req == None:
            tra_sorted =str("∅")
            logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
        else:
            tra_match = {}
            for i in range(len(self.tra_req)-1):
                tra_match[i] = self.filter_compatible(self.tra_req[i], self.model_indexes["tra"])
                if tra_match[i] == None:
                    tra_match[i] == query
            try:
                if tra_match[0] == []:
                    logger.debug(f"No external text encoder found compatible with {query}.", exc_info=True)
                    tra_sorted = {}
                else:
                    for each in tra_match.keys():
                        tra_sorted[tra_match[each][0][0][1]] = tra_match[each][0][1]
            except IndexError as error_log:
                logger.debug(f"Error when returning encoder for {query} : {error_log}.", exc_info=True)
                tra_sorted = {}
        vae_sorted = self.filter_compatible(query, self.model_indexes["vae"])
        lora_sorted = self.filter_compatible(query, self.model_indexes["lor"])
        if len(lora_sorted) >1:
            lora_sorted = dict(lora_sorted)
        else:
            lora_sorted = []
        if vae_sorted == []:
            vae_sorted =str("∅")
            print(query)
            logger.debug(f"No external VAE found compatible with {query}.", exc_info=True)
        if lora_sorted == []:
            lora_sorted =str("∅")
            logger.debug(f"No compatible LoRA found for {query}.", exc_info=True)

        return vae_sorted, tra_sorted, lora_sorted

    def fetch_refiner(self):
        self.dif_index = config.get_default("index", "DIF")
        for key, value in self.dif_index.items():
            if key == "STA-XR":
                return value
        return "∅"

    #within a dict of models of the same type, match model code & sort by file size
    def filter_compatible(self, query, index):
        pack = defaultdict(dict)
        if index.items():
            for k, v in index.items():
                for code in v.keys():
                    if query in code:
                        pack[k, code] = v[code]
                        
            sort = sorted(pack.items(), key=lambda item: item[1])
            return sort
        else:
            logger.debug("Compatible models not found")
            return "∅"


# import torch
# import re
# import sys

# from math import isclose
# from functools import reduce
# import hashlib

# from tqdm.auto import tqdm

# from sdbx import logger
# from sdbx.config import config

# class Domain:
#     """Represents a top-level domain like nn, info, or dev."""

#     def __init__(self, domain_name):
#         self.domain_name = domain_name
#         self.architectures = {}  # Architecture objects

#     def add_architecture(self, architecture_name, architecture_obj):
#         self.architectures[architecture_name] = architecture_obj

#     def flatten(self, prefix):
#         """Flattens the block format to a dict."""
#         flat_dict = {}
#         for arc_name, arc_obj in self.architectures.items():
#             path = f"{prefix}.{arc_name}"
#             flat_dict.update(arc_obj.flatten(path))
#         return flat_dict


#     class Architecture:
#         """Represents model architecture like sdxl, flux."""

#         def __init__(self, architecture):
#             self.architecture = architecture
#             self.components = {}  # Component objects

#         def add_component(self, component_name, component_obj):
#             self.components[component_name] = component_obj

#         def flatten(self, prefix):
#             """Flattens the architecture to a dict."""
#             flat_dict = {}
#             for comp_name, comp_obj in self.components.items():
#                 path = f"{prefix}.{comp_name}"
#                 flat_dict[path] = comp_obj.to_dict()
#             return flat_dict

#         class Component:
#             """Represents individual model components like vae, lora, unet."""
#             def __init__(self, component_name, **kwargs):
#                 self.component_name = component_name

#                 if 'dtype' in kwargs: self.dtype = kwargs['dtype']
#                 if 'file_size' in kwargs: self.file_size = kwargs['file_size']
#                 if 'distinction' in kwargs: self.distinction = kwargs['distinction']

#             def to_dict(self):
#                 """Serializes the Component object to a dictionary."""
#                 return {
#                     'component_name': self.component_name,
#                     'dtype': self.dtype,
#                     'file_size': self.file_size,
#                     'distinction': self.distinction
#                 }


# class MatchMetadata:

#     def extract_tensor_data(self, source_data_item, id_values):
#         """
#         Extracts shape and key data from the source data.
#         This would extract whatever additional information is needed when a match is found.
#         """
#         TENSOR_TOLERANCE = 4e-2
#         search_items = ["dtype", "shape"]
#         for field_name in search_items:
#             field_value = source_data_item.get(field_name)
#             if field_value:
#                 if isinstance(field_value, list):
#                     field_value = str(field_value)  # We only need the first two numbers of 'shape'
#                 if field_value not in id_values.get(field_name,""):
#                     id_values[field_name] = " ".join([id_values.get(field_name, ""),field_value]).lstrip() # Prevent data duplication

#         return {
#             "tensors": id_values.get("tensors", 0),
#             'shape': id_values.get('shape', 0),
#         }

#     def find_matching_metadata(self, known_values, source_data, id_values, depth=[]):
#         """
#         Recursively traverse the criteria and source_data to check for matches.
#         Track comparisons with id_values, using a list to track the recursion depth.

#         known_values: the original hierarchy of known values from a .json file
#         source_data: the original model state dict
#         id_values: information matching our needs extracted from the source data
#         depth: current level inside the dict
#         """

#         id_values = id_values

#         # Get the dict position indicated in depth
#         get_nested = lambda d, keys: reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys, d)
#         # Return the previous position indicated in depth
#         backtrack_depth = lambda depth: depth[:-1] if depth and depth[-1] in ["block_names", "tensors", "shape", "file_size", "hash"] else depth

#         def advance_depth(depth: list, lateral: bool = False) -> list:
#             """
#             Attempts to advance through the tuning dict laterally (to the next key at the same level),
#             failing which retraces vertically (first to the parent level, then the next key at parent level).
#             """
#             if not depth:
#                 return None  # Stop if we've reached the root or there's no further depth
#             parent_dict = get_nested(known_values, depth[:-1]) # Prior depth

#             if not isinstance(parent_dict, dict):    # We look for dicts, and no other types
#                 return None  # Invalid state if we can't get the parent dict

#             parent_keys = list(parent_dict.keys())  #  Keys from above
#             previous_depth = depth[-1] # Current level

#             if previous_depth in parent_keys: # Lateral movement check
#                 current_index = parent_keys.index(previous_depth)

#                 if current_index + 1 < len(parent_keys): #Lateral/next movement, same level
#                     new_depth = depth[:-1]  # Get the parent depth
#                     new_depth.append(parent_keys[current_index + 1])  # Add the next key in sequence
#                     return new_depth

#             if len(depth) > 1: # If no lateral movement is possible, try  vertical/backtracking if there is more than one level
#                 return advance_depth(depth[:-1])  # Move to parent and retry

#             return None  # Traversal complete

#         criteria = get_nested(known_values, depth)
#         if criteria is None:  # Cannot advance, stop
#             return id_values

#         if isinstance(criteria, str): criteria = [criteria]
#         if isinstance(criteria, dict):
#             for name in criteria: # Descend dictionary structure
#                 depth.append(name) # Append the current name to depth list
#                 self.find_matching_metadata(known_values, source_data, id_values, depth)
#                 if depth is None:  # Cannot advance, stop
#                     return id_values
#                 else:
#                     depth = backtrack_depth(depth)
#                     current_depth = get_nested(known_values, depth)
#                     if current_depth[-1] ==
#                         if len(current_depth) == id_values.get(depth[-1],0):
#                             id_values.get("type", set()).add(depth[-1])

#                         known_values[next(iter(known_values), "nn")].pop(depth[-1])
#                     advance_depth(depth)
#                     self.find_matching_metadata(known_values, source_data, id_values, depth)
#             return id_values

#         elif isinstance(criteria, list): # when fed correct datatype, we check for matches
#             for checklist in criteria:
#                 if not isinstance(checklist, list): checklist = [checklist]  # normalize scalar to list
#                 for list_entry in checklist: # the entries to match
#                     if depth[-1] == "hash":
#                          id_values["hash"] = hashlib.sha256(open(id_values["file_name"],'rb').read()).hexdigest()
#                     list_entry = str(list_entry)
#                     if list_entry.startswith("r'"): # Regex conversion
#                         expression = (list_entry
#                             .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
#                             .replace(".", r"\.")    # Escape literal dots with '\.'
#                             .strip("r'")            # Strip the 'r' and quotes from the string
#                         )
#                         regex_entry = re.compile(expression)
#                         match = next((regex_entry.search(k) for k in source_data), False)
#                     else:
#                         match = next((k for k in source_data if list_entry in k), False)
#                     if match: # Found a match, based on keys
#                         previous_depth = depth[-1]
#                         depth = backtrack_depth(depth)
#                         found = depth[-1] if depth else "unknown"    # if theres no header or other circumstances
#                         id_values[found] = id_values.get(found, 0) + 1

#                         shape_key_data = self.extract_tensor_data(source_data[match], id_values)
#                         id_values.update(shape_key_data)

#                         depth.append(previous_depth) #if length depth = 2
#                         depth = advance_depth(depth, lateral=True)
#                         if depth is None:  # Cannot advance, stop
#                             return id_values
#                         self.find_matching_metadata(known_values, source_data, id_values, depth)  # Recurse


#             return id_values


# class BlockIndex:
#     # 重みを確認するモデルファイル
#     def main(self, file_name: str, path: str):

#         self.id_values = defaultdict(dict)
#         file_suffix = Path(file_name).suffix
#         if file_suffix == "": return
#         self.id_values["file_name"] = file_name
#         self.id_values["extension"] = Path(file_name).suffix.lower() # extension is metadata
#         model_header = defaultdict(dict)

#         # Process file by method indicated by extension, usually struct unpacking, except for pt files which are memmap
#         if self.id_values["extension"] in [".safetensors", ".sft"]: model_header: dict = self.__unsafetensors(file_name, self.id_values["extension"])
#         elif self.id_values["extension"] == ".gguf": model_header: dict = self.__ungguf(file_name, self.id_values["extension"])
#         elif self.id_values["extension"] in [".pt", ".pth"]: model_header: dict = self.__unpickle(file_name, self.id_values["extension"])

#         if model_header:
#             self.neural_net = Domain("nn") #create the domain only when we know its a model

#             self.MODEL_FORMAT = config.get_default("tuning","formats")
#             self.id_values["tensors"] = len(model_header)
#             instance = MatchMetadata()
#             self.id_values = instance.find_matching_metadata(known_values=self.MODEL_FORMAT, source_data=model_header, id_values=self.id_values)

#             self._pretty_output(file_name)
#             filename = os.path.join(path, os.path.basename(file_name) + ".json")
#             with open(filename, "w", encoding="UTF-8") as index: # todo: make 'a' type before release
#                 data = self.id_values | model_header
#                 json.dump(data ,index, ensure_ascii=False, indent=4, sort_keys=False)

#     def error_handler(self, kind:str, error_log:str, obj_name:str=None, error_source:str=None):
#         if kind == "retry":
#             self.id_progress("Error reading metadata, switching read method")
#         elif kind == "fail":
#             self.id_progress("Metadata read attempts exhasted for:'", obj_name)
#         logger.debug(f"Could not read : '{obj_name}' In {error_source}: {error_log}", exc_info=True)
#         return

# #SAFETENSORS
#     def __unsafetensors(self, file_name:str, extension: str):
#         self.id_values["extension"] = "safetensors"
#         self.id_values["file_size"] = os.path.getsize(file_name)
#         with open(file_path, 'rb') as file:
#             try:
#                 first_8_bytes    = file.read(8)
#                 length_of_header = struct.unpack('<Q', first_8_bytes)[0]
#                 header_bytes     = file.read(length_of_header)
#                 header           = json.loads(header_bytes.decode('utf-8'))
#                 if header.get("__metadata__",0 ) != 0:  # we want to remove this metadata so its not counted as tensors
#                     header.pop("__metadata__")  # it is usually empty on safetensors ._.
#                 return header
#             except Exception as error_log:  #couldn't open file
#                 self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension)

# # GGUF
#     def __ungguf(self, file_name:str, extension:str):
#         self.id_values["file_size"] = os.path.getsize(file_name) # how big will be important for memory management
#         file_data = defaultdict(dict)
#         from llama_cpp import Llama
#         try:
#             with open(file_name, "rb") as file:
#                 magic = file.read(4)
#                 if magic != b"GGUF":
#                     logger.debug(f"Invalid GGUF magic number in '{file_name}'") # uh uh uh, you didn't say the magic word
#                     return
#                 version = struct.unpack("<I", file.read(4))[0]
#                 if version < 2:
#                     logger.debug(f"Unsupported GGUF version {version} in '{file_name}'")
#                     return
#             parser                  = Llama(model_path=file_name, vocab_only=True, verbose=False) #  fails image quants, but dramatically faster vs ggufreader
#             arch                    = parser.metadata.get("general.architecture") # with gguf we can directly request the model name but it isnt always written in full
#             name                    = parser.metadata.get("general.name") # sometimes this field is better than arch
#             self.id_values["name"]  = name if name is not None else arch
#             self.id_values["dtype"] = parser.scores.dtype.name #outputs as full name eg: 'float32 rather than f32'
#             return # todo: handle none return better
#         except ValueError as error_log:
#             self.error_handler(kind="retry", error_log=error_log, obj_name=file_name, error_source=extension) # the aforementioned failing
#         from gguf import GGUFReader
#         try: # method using gguf library, better for LDM conversions
#             reader                  = GGUFReader(file_name, 'r')
#             self.id_values["dtype"] = reader.data.dtype.name # get dtype from metadata
#             arch                    = reader.fields["general.architecture"] # model type category, usually prevents the need  toblock scan for llms
#             self.id_values["name"]  = str(bytes(arch.parts[arch.data[0]]), encoding='utf-8') # retrieve model name from the dict data
#             if len(arch.types) > 1:
#                 self.id_values["name"] = arch.types #if we get a result, save it
#             for tensor in reader.tensors:
#                 file_data[str(tensor.name)] = {"shape": str(tensor.shape), "dtype": str(tensor.tensor_type.name)} # create dict similar to safetensors/pt results
#             return file_data
#         except ValueError as error_log:
#             self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension) # >:V

# # PICKLETENSOR FILE
#     def __unpickle(self, file_name:str, extension:str):
#         self.id_values["file_size"] = os.path.getsize(file_name) #
#         import mmap
#         import pickle
#         try:
#             return torch.load(file_name, map_location="cpu") #this method seems outdated
#         except TypeError as error_log:
#             self.error_handler(kind="retry", error_log=error_log, obj_name=file_name, error_source=extension)
#             try:
#                 with open(file_name, "r+b") as file_obj:
#                     mm = mmap.mmap(file_obj.fileno(), 0)
#                     return pickle.loads(memoryview(mm))
#             except Exception as error_log: #throws a _pickle error (so salty...)
#                 self.error_handler(kind="fail", error_log=error_log, obj_name=file_name, error_source=extension)

#     def _pretty_output(self, file_name): #pretty printer
#         print_values = self.id_values.copy()
#         if (k := next(iter(print_values), None)) is not None:
#             print_values.pop(k)  # Only pop if a valid key is found
#         key_value_length = len(print_values)  # number of items detected in the scan
#         info_format      = "{:<5} | " * key_value_length # shrink print columns to data width
#         header_keys      = tuple(print_values) # use to create table
#         horizontal_bar   = ("  " + "-" * (10*key_value_length)) # horizontal divider of arbitrary length. could use shutil to dynamically create but eh. already overkill
#         formatted_data   = tuple(print_values.values()) # data extracted from the scan
#         return self.id_progress(self.id_values.get("file_name", None), info_format.format(*header_keys), horizontal_bar, info_format.format(*formatted_data)) #send to print function

#     def id_progress(self, *formatted_data):
#         sys.stdout.write("\033[F" * len(formatted_data))  # ANSI escape codes to move the cursor up 3 lines
#         for line_data in formatted_data:
#             sys.stdout.write(" " * 175 + "\x1b[1K\r")
#             sys.stdout.write(f"{line_data}\r\n")  # Print the lines
#         sys.stdout.flush()              # Empty output buffer to ensure the changes are shown

# if __name__ == "__main__":
#     file = config.get_path("models.dev")
#     blocks = BlockIndex()
#     save_location = os.path.join(config.get_path("models.dev"),"metadata")
#     if Path(file).is_dir() == True:
#         path_data = os.listdir(file)
#         print("\n\n\n\n")
#         for each_file in tqdm(path_data, total=len(path_data), position=0, leave=True):
#             file_path = os.path.join(file,each_file)
#             blocks.main(file_path, save_location)
#     else:
#         blocks.main(file, save_location)


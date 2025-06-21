import os
import shutil
import argparse
import configparser
import logging
import json
from datetime import datetime
from PIL import Image, ImageStat
import numpy as np
import cv2

#Set up logging for easy debugging and tracking
log_filename = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """
    Loads configuration from a file if it exists, otherwise uses default values.
    This helps keep paths flexible and easy to change without editing the script.
    """
    config = {
        'BASE_PROJECT_PATH': 'C:\\Users\\Mohamed Sameh\\OneDrive\\IT_Task',
        'SOURCE_DIR': 'Photos4Testing',
        'WEBSITE_OUTPUT_DIR': 'OP_PhotosTested',
        'PUBLICATION_OUTPUT_DIR': 'OP_PublicationPhotosTested'
    }
    
    #Try to read from a config file if present
    if os.path.exists('photo_validator_config.ini'):
        try:
            cfg = configparser.ConfigParser()
            cfg.read('photo_validator_config.ini')
            if 'Paths' in cfg:
                for key in config:
                    if key in cfg['Paths']:
                        config[key] = cfg['Paths'][key]
            logger.info("Configuration loaded from file")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    #Build absolute paths for all directories
    config['SOURCE_DIR'] = os.path.join(config['BASE_PROJECT_PATH'], config['SOURCE_DIR'])
    config['WEBSITE_OUTPUT_DIR'] = os.path.join(config['BASE_PROJECT_PATH'], config['WEBSITE_OUTPUT_DIR'])
    config['PUBLICATION_OUTPUT_DIR'] = os.path.join(config['BASE_PROJECT_PATH'], config['PUBLICATION_OUTPUT_DIR'])
    
    return config

def parse_args():
    """
    Handles command-line arguments so users can override config values easily.
    """
    parser = argparse.ArgumentParser(description='Validate photos against university specifications')
    parser.add_argument('--source', help='Source directory containing photos to validate')
    parser.add_argument('--web-output', help='Output directory for valid website photos')
    parser.add_argument('--pub-output', help='Output directory for valid publication photos')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with extra logging')
    return parser.parse_args()

#Define image requirements for website and publication use cases as in the required-specificatons pdf.
SPECS = {
    'website': {
        'slider': {
            'width': 1920,
            'height': 705,
            'dpi': (72, 72),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg'],
            'orientation': 'landscape'
        },
        'gallery': {
            'width': 800,
            'height': 600,
            'dpi': (72, 72),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg'],
            'orientation': 'landscape'
        }
    },
    'publication': {
        'digital_a4': {
            'width': 2480,
            'height': 3508,
            'dpi_range': (150, 200),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'PNG', 'png'],
            'color_mode': 'RGB'
        },
        'digital_b5': {
            'width': 2069,
            'height': 2953,
            'dpi_range': (150, 200),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'PNG', 'png'],
            'color_mode': 'RGB'
        },
        'print_a4': {
            'width': 2480,
            'height': 3508,
            'dpi': (300, 300),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'TIFF', 'tiff', 'tif', 'TIF'],
            'color_mode': 'CMYK'
        },
        'print_b5': {
            'width': 2079,
            'height': 2953,
            'dpi': (300, 300),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'TIFF', 'tiff', 'tif', 'TIF'],
            'color_mode': 'CMYK'
        }
    }
}

#Advanced image analysis functions

def has_borders(img, threshold=10):
    """
    Checks if the image has borders or frames by looking for low-variance edges.
    This is a quick way to spot obvious frames or borders.
    """
    try:
        img_array = np.array(img)
        #grab some pixels from each edge
        top_edge = img_array[0:5, :, :]
        bottom_edge = img_array[-5:, :, :]
        left_edge = img_array[:, 0:5, :]
        right_edge = img_array[:, -5:, :]
        #If edge is too uniform -> may be a border
        edge_std = [
            np.std(top_edge), 
            np.std(bottom_edge),
            np.std(left_edge),
            np.std(right_edge)
        ]
        if any(std < threshold for std in edge_std):
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking for borders: {e}")
        return False

def has_watermark_or_logo(img, threshold=0.15):
    """
    Looks for watermarks or logos by checking for unusual edge patterns.
    This is a basic check and may not catch everything.
    threshold is the edge density to consider it a watermark.
    A higher threshold means more edges are needed to flag as a watermark.
    """
    try:
        #convert to grayscale for easier edge detection
        if img.mode != 'L':
            gray_img = img.convert('L')
        else:
            gray_img = img
        img_array = np.array(gray_img)
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])
        #if lot of edges but not too many -> may be a watermark
        if edge_density > threshold and edge_density < 0.5:
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking for watermarks: {e}")
        return False

def has_text_overlay(img):
    """
    Tries to spot text overlays by finding lots of small, rectangular contours.
    This is a rough way to catch obvious text on images.
    """
    try:
        if img.mode != 'L':
            gray_img = img.convert('L')
        else:
            gray_img = img
        img_array = np.array(gray_img)
        _, binary = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_like_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.1 < aspect_ratio < 15 and 10 < w < 300 and 10 < h < 100:
                text_like_contours += 1
        #if there are several text-like shapes, mark/flag as possible text overlay
        if text_like_contours > 5:
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking for text overlay: {e}")
        return False

def has_unnatural_colors(img, saturation_threshold=180, contrast_threshold=80):
    """
    Checks for unnatural colors or heavy filtering by looking at saturation and contrast.
    This helps catch images that have been over-edited.
    """
    try:
        if img.mode != 'RGB':
            rgb_img = img.convert('RGB')
        else:
            rgb_img = img
        stat = ImageStat.Stat(rgb_img)
        r_mean, g_mean, b_mean = stat.mean
        r_std, g_std, b_std = stat.stddev
        img_array = np.array(rgb_img)
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv_img[:, :, 1]
        avg_saturation = np.mean(saturation)
        contrast = max(r_std, g_std, b_std)
        # Flag if either saturation or contrast is too high
        if avg_saturation > saturation_threshold or contrast > contrast_threshold:
            return True
        # Also check if one color dominates too much
        h_hist = np.histogram(hsv_img[:, :, 0], bins=36)[0]
        h_hist = h_hist / np.sum(h_hist)
        if np.max(h_hist) > 0.4:
            return True
        return False
    except Exception as e:
        logger.warning(f"Error checking for unnatural colors: {e}")
        return False

def check_photo(image_path, rules, debug=False):
    """
    Checks a single photo against a set of rules.
    Returns (True, "Valid") if all checks pass, otherwise (False, reason).
    """
    try:
        if debug:
            logger.info(f"DEBUG: Opening image {image_path}")
        with Image.open(image_path) as img:
            if debug:
                logger.info(f"DEBUG: Image opened successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
            # Check file format
            if 'formats' in rules:
                if debug:
                    logger.info(f"DEBUG: Checking format. Image format: {img.format}, Expected formats: {rules['formats']}")
                if img.format not in rules['formats']:
                    return (False, f"Invalid format: is {img.format}, should be one of {rules['formats']}")
            #Check dimensions, allow a small tolerance
            if 'width' in rules and 'height' in rules:
                width_tolerance = rules['width'] * 0.02
                height_tolerance = rules['height'] * 0.02
                if debug:
                    logger.info(f"DEBUG: Checking dimensions. Image size: {img.size}, Expected: ({rules['width']}, {rules['height']}), Tolerance: ±{width_tolerance}/{height_tolerance}")
                if abs(img.width - rules['width']) > width_tolerance or abs(img.height - rules['height']) > height_tolerance:
                    return (False, f"Invalid dimensions: is {img.size}, should be ({rules['width']}, {rules['height']})")
            #Check DPI if available
            dpi = img.info.get('dpi')
            if debug:
                logger.info(f"DEBUG: DPI info in image: {dpi}")
            if 'dpi' in rules:
                if dpi is None:
                    logger.warning(f"Image {os.path.basename(image_path)} is missing DPI information, assuming default")
                else:
                    if dpi != rules['dpi']:
                        return (False, f"Invalid DPI: is {dpi}, should be {rules['dpi']}")
            #Check DPI range for digital publication images
            if 'dpi_range' in rules and dpi is not None:
                min_dpi, max_dpi = rules['dpi_range']
                if not (min_dpi <= dpi[0] <= max_dpi and min_dpi <= dpi[1] <= max_dpi):
                    return (False, f"Invalid DPI: is {dpi}, should be between {min_dpi} and {max_dpi}")
            #Check color mode (RGB or CMYK)
            if 'color_mode' in rules:
                if debug:
                    logger.info(f"DEBUG: Checking color mode. Image mode: {img.mode}, Expected: {rules['color_mode']}")
                if img.mode != rules['color_mode']:
                    #Allow RGBA for RGB requirements
                    if not (rules['color_mode'] == 'RGB' and img.mode == 'RGBA'):
                        return (False, f"Invalid color mode: is {img.mode}, should be {rules['color_mode']}")
            #Check orientation for website images
            if 'orientation' in rules and rules['orientation'] == 'landscape':
                if debug:
                    logger.info(f"DEBUG: Checking orientation. Width: {img.width}, Height: {img.height}")
                if img.width <= img.height:
                    return (False, "Invalid orientation: should be landscape (width > height)")
            #Run advanced checks only if debug is enabled (for speed)
            if debug:
                logger.info(f"DEBUG: Running advanced image analysis")
                if has_borders(img):
                    return (False, "Image has borders or frames")
                if has_watermark_or_logo(img):
                    return (False, "Image appears to have watermarks or logos")
                if has_text_overlay(img):
                    return (False, "Image has text overlays")
                if has_unnatural_colors(img):
                    return (False, "Image has unnatural colors or excessive filtering/effects")
    except Exception as e:
        logger.error(f"Could not read or process file {image_path}. Error: {e}")
        return (False, f"Could not process file. Error: {e}")
    #All checks passed ;)
    return (True, "Valid")

def process_and_copy_files(category_name, ruleset, source_folder, output_folder, debug=False):
    """
    Goes through all files in the source folder, checks them against the rules,
    and copies valid images to the output folder.
    """
    logger.info(f"{'='*20}\nProcessing for: {category_name.upper()}\n{'='*20}")
    logger.info(f"Source folder: {source_folder}")
    logger.info(f"Output folder: {output_folder}")

    #make sure the output directory exists
    if not os.path.exists(output_folder):
        logger.info(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)

    #collect all supported file extensions for this category
    supported_extensions = set()
    for spec in ruleset.values():
        for fmt in spec.get('formats', []):
            supported_extensions.add(f".{fmt.lower()}")
    if debug:
        logger.info(f"DEBUG: Looking for files with extensions: {supported_extensions}")
    try:
        all_files = os.listdir(source_folder)
        if debug:
            logger.info(f"DEBUG: Found {len(all_files)} total files in {source_folder}")
            logger.info(f"DEBUG: Files in directory: {all_files}")
    except Exception as e:
        logger.error(f"Error listing directory {source_folder}: {e}")
        all_files = []

    copied_files = set()
    results = {
        'total_files_processed': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'validation_details': []
    }

    for filename in all_files:
        #only process files with supported image extensions
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in supported_extensions:
            if debug:
                logger.info(f"DEBUG: Skipping {filename} - extension {file_ext} not in {supported_extensions}")
            continue
        if debug:
            logger.info(f"DEBUG: Found image file: {filename}")
        file_path = os.path.join(source_folder, filename)
        if not os.path.isfile(file_path):
            if debug:
                logger.info(f"DEBUG: Skipping {filename} - not a file")
            continue
        is_valid_for_any_rule = False
        file_result = {
            'filename': filename,
            'category': category_name,
            'valid': False,
            'matching_rule': None,
            'reasons_for_failure': []
        }
        results['total_files_processed'] += 1
        logger.info(f"\n--- Checking: {filename} ---")
        #try each rule; if any passes, the image is valid for this category
        for rule_name, rule_details in ruleset.items():
            is_valid, reason = check_photo(file_path, rule_details, debug)
            if is_valid:
                logger.info(f"  [PASS] Matches '{rule_name}' criteria.")
                file_result['valid'] = True
                file_result['matching_rule'] = rule_name
                # Only copy the file once, even if it matches multiple rules
                if filename not in copied_files:
                    try:
                        shutil.copy2(file_path, os.path.join(output_folder, filename))
                        logger.info(f"  [ACTION] Copied to {output_folder}")
                        copied_files.add(filename)
                    except Exception as e:
                        logger.error(f"  [ERROR] Could not copy file: {e}")
                is_valid_for_any_rule = True
                results['valid_files'] += 1
                break
            else:
                logger.info(f"  [FAIL] Does not match '{rule_name}'. Reason: {reason}")
                file_result['reasons_for_failure'].append(f"{rule_name}: {reason}")
        if not is_valid_for_any_rule:
            logger.info(f"  [RESULT] {filename} is not valid for any {category_name} specifications.")
            results['invalid_files'] += 1
        results['validation_details'].append(file_result)
    return results

def generate_validation_report(website_results, publication_results, output_path):
    """
    Creates a detailed validation report as a JSON file and prints a summary to the console.
    """
    combined_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_files_processed': website_results['total_files_processed'],
            'website_valid_files': website_results['valid_files'],
            'website_invalid_files': website_results['invalid_files'],
            'publication_valid_files': publication_results['valid_files'],
            'publication_invalid_files': publication_results['invalid_files'],
        },
        'website_results': website_results,
        'publication_results': publication_results
    }
    #calculate and add success rates for both categories
    if website_results['total_files_processed'] > 0:
        combined_results['summary']['website_success_rate'] = round(
            website_results['valid_files'] / website_results['total_files_processed'] * 100, 2
        )
    if publication_results['total_files_processed'] > 0:
        combined_results['summary']['publication_success_rate'] = round(
            publication_results['valid_files'] / publication_results['total_files_processed'] * 100, 2
        )
    #save the full report to a file
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    #print a quick summary for the user
    logger.info("\n\n" + "="*50)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files processed: {combined_results['summary']['total_files_processed']}")
    logger.info(f"Website valid files: {combined_results['summary']['website_valid_files']}")
    logger.info(f"Publication valid files: {combined_results['summary']['publication_valid_files']}")
    if 'website_success_rate' in combined_results['summary']:
        logger.info(f"Website validation success rate: {combined_results['summary']['website_success_rate']}%")
    if 'publication_success_rate' in combined_results['summary']:
        logger.info(f"Publication validation success rate: {combined_results['summary']['publication_success_rate']}%")
    logger.info(f"Detailed report saved to: {output_path}")
    logger.info("="*50)
    return combined_results

def main():
    """
    This is the main entry point for the script.
    Loads config, parses arguments, processes images, and generates reports.
    """
    logger.info("Starting photo validation script...")
    #load configuration (from file or defaults)
    config = load_config()
    #parse command-line args
    args = parse_args()
    debug_mode = args.debug
    #allow command-line overrides for any path
    if args.source:
        config['SOURCE_DIR'] = args.source
    if args.web_output:
        config['WEBSITE_OUTPUT_DIR'] = args.web_output
    if args.pub_output:
        config['PUBLICATION_OUTPUT_DIR'] = args.pub_output
    #make sure the source directory exists before proceeding
    if not os.path.isdir(config['SOURCE_DIR']):
        logger.error(f"Error: The source directory does not exist. Please check the path:")
        logger.error(f"'{config['SOURCE_DIR']}'")
        logger.error("Update the configuration or provide a valid path via command-line arguments.")
        return
    if debug_mode:
        logger.info(f"DEBUG MODE ENABLED")
        logger.info(f"SOURCE_DIR: {config['SOURCE_DIR']}")
        try:
            files = os.listdir(config['SOURCE_DIR'])
            logger.info(f"Files in source directory: {files}")
        except Exception as e:
            logger.error(f"Error listing source directory: {e}")
    #validate and copy website images
    website_results = process_and_copy_files('website', SPECS['website'], 
                                             config['SOURCE_DIR'], 
                                             config['WEBSITE_OUTPUT_DIR'],
                                             debug=debug_mode)
    #validate and copy publication images
    publication_results = process_and_copy_files('publication', SPECS['publication'], 
                                                 config['SOURCE_DIR'], 
                                                 config['PUBLICATION_OUTPUT_DIR'],
                                                 debug=debug_mode)
    #save a detailed validation report
    report_path = os.path.join(os.path.dirname(config['SOURCE_DIR']), 
                              f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    generate_validation_report(website_results, publication_results, report_path)
    logger.info("Script finished. Check the output folders for valid photos and see the reports for details.")

def create_sample_config():
    """
    Creates a sample configuration file if one doesn't already exist.
    This makes it easier for new users to get started.
    """
    if not os.path.exists('photo_validator_config.ini'):
        config = configparser.ConfigParser()
        config['Paths'] = {
            'BASE_PROJECT_PATH': 'C:\\Users\\Mohamed Sameh\\OneDrive\\IT_Task',
            'SOURCE_DIR': 'Photos4Testing',
            'WEBSITE_OUTPUT_DIR': 'OP_PhotosTested',
            'PUBLICATION_OUTPUT_DIR': 'OP_PublicationPhotosTested'
        }
        with open('photo_validator_config.ini', 'w') as f:
            config.write(f)
        print("Created sample configuration file: photo_validator_config.ini")

#run everything if script is executed directly
if __name__ == "__main__":
    create_sample_config()
    main()
import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {v: k for k, v in dict_char_to_int.items()}  # Inverse mapping

def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
        
        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                if 'car' in car_data and 'license_plate' in car_data:
                    plate_data = car_data['license_plate']
                    if 'text' in plate_data:
                        f.write(f"{frame_nmr},{car_id},"
                                f"[{','.join(map(str, car_data['car']['bbox']))}],"
                                f"[{','.join(map(str, plate_data['bbox']))}],"
                                f"{plate_data['bbox_score']},{plate_data['text']},{plate_data['text_score']}\n")

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    """
    if len(text) != 7:
        return False
    letters_indices = {0, 1, 4, 5, 6}
    numbers_indices = {2, 3}
    return all(
        (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char) if i in letters_indices else
        (text[i] in string.digits or text[i] in dict_char_to_int) for i in range(7)
    )

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    """
    mapping = {**dict_int_to_char, **dict_char_to_int}  # Merge both dictionaries
    return ''.join(mapping.get(char, char) for char in text)

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    """
    detections = reader.readtext(license_plate_crop)
    
    for _, text, score in detections:
        formatted_text = text.upper().replace(' ', '')
        if license_complies_format(formatted_text):
            return format_license(formatted_text), score
    
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, *_ = license_plate
    
    for track in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2 = track.to_tlbr()
        
        if xcar1 < x1 < xcar2 and ycar1 < y1 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, track.track_id
    
    return -1, -1, -1, -1, -1
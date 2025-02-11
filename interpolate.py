import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split(','))) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split(','))) for row in data])
    
    row_lookup = {(int(row['frame_nmr']), int(float(row['car_id']))): row for row in data}
    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        car_indices = np.where(car_ids == car_id)[0]
        car_frame_numbers = frame_numbers[car_indices]
        sort_order = np.argsort(car_frame_numbers)
        car_indices = car_indices[sort_order]
        car_frame_numbers = frame_numbers[car_indices]
        
        car_bbox_data = car_bboxes[car_indices]
        lp_bbox_data = license_plate_bboxes[car_indices]
        
        interp_car_bboxes = [car_bbox_data[0]]
        interp_lp_bboxes = [lp_bbox_data[0]]

        for i in range(1, len(car_frame_numbers)):
            current_frame, prev_frame = car_frame_numbers[i], car_frame_numbers[i - 1]
            current_car_bbox, prev_car_bbox = car_bbox_data[i], interp_car_bboxes[-1]
            current_lp_bbox, prev_lp_bbox = lp_bbox_data[i], interp_lp_bboxes[-1]
            
            if current_frame - prev_frame > 1:
                frames_gap = current_frame - prev_frame
                x = np.array([prev_frame, current_frame])
                x_new = np.linspace(prev_frame, current_frame, num=frames_gap, endpoint=False)[1:]
                
                interp_func_car = interp1d(x, np.vstack((prev_car_bbox, current_car_bbox)), axis=0, kind='linear')
                new_car_bboxes = interp_func_car(x_new)
                
                interp_func_lp = interp1d(x, np.vstack((prev_lp_bbox, current_lp_bbox)), axis=0, kind='linear')
                new_lp_bboxes = interp_func_lp(x_new)
                
                interp_car_bboxes.extend(new_car_bboxes)
                interp_lp_bboxes.extend(new_lp_bboxes)
            
            interp_car_bboxes.append(current_car_bbox)
            interp_lp_bboxes.append(current_lp_bbox)
        
        first_frame = car_frame_numbers[0]
        num_total = len(interp_car_bboxes)
        for idx in range(num_total):
            frame = first_frame + idx
            new_row = {
                'frame_nmr': str(frame),
                'car_id': str(car_id),
                'car_bbox': '[' + ','.join(map(str, interp_car_bboxes[idx])) + ']',
                'license_plate_bbox': '[' + ','.join(map(str, interp_lp_bboxes[idx])) + ']'
            }
            
            if frame not in car_frame_numbers:
                new_row.update({'license_plate_bbox_score': '0', 'license_number': '0', 'license_number_score': '0'})
            else:
                orig = row_lookup.get((frame, car_id), {})
                new_row.update({
                    'license_plate_bbox_score': orig.get('license_plate_bbox_score', '0'),
                    'license_number': orig.get('license_number', '0'),
                    'license_number_score': orig.get('license_number_score', '0')
                })
            interpolated_data.append(new_row)
    
    return interpolated_data

if __name__ == '__main__':
    with open('test.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    interpolated_data = interpolate_bounding_boxes(data)
    
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open('test_interpolated1.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

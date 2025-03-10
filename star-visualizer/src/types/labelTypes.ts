export interface BoundingBox {
    class_id: number;
    x_center: number;
    y_center: number;
    width: number;
    height: number;
}

export interface ImageLabel {
    image_name: string;
    bounding_boxes: BoundingBox[];
}

export interface LabelsData {
    image_names: string[];
    labels: ImageLabel[];
} 
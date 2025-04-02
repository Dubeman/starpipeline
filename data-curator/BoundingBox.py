class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    @classmethod
    def from_string(cls, label_string):
        parts = label_string.split()
        return cls(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, label_dict):
        return cls(label_dict["class_id"], label_dict["x_center"], label_dict["y_center"], label_dict["width"], label_dict["height"])
    
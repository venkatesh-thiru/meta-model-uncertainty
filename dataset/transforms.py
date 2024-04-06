import torchvision.transforms.functional as F


class Resize_with_pad:
    def __init__(self, w=1024, h=1024):
        self.w = w
        self.h = h

    def __call__(self, image):
        w_1, h_1 = image.shape[-1], image.shape[-2]
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1


        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w], antialias=None)

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w],antialias=None)

        else:
            return F.resize(image, [self.h, self.w], antialias=None)


class ImageRecord:
  def __init__(self, dataset, fn, illum, mcc_coord, img, extras=None):
    self.dataset = dataset
    self.fn = fn
    self.illum = illum
    self.mcc_coord = mcc_coord
    # BRG images
    self.img = img
    self.extras = extras

  def __repr__(self):
    return '[%s, %s, (%f, %f, %f)]' % (self.dataset, self.fn, self.illum[0],
                                       self.illum[1], self.illum[2])

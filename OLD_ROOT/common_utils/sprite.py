#utils.py
import glob
from PIL import Image
import cStringIO as StringIO

def select_frames(num):
    # print "Selecting from num", num
    items = [num/2] + [int(i * (num)/10.0) for i in range(10)]
    return items

def create_sprite(frame_path, width=200, num_frames=10):
    """
        # Supposed to create middle_preview and sprite image
        # Stores them in the same folder structure
        # Creates middle_preview
    """
    print frame_path + "/*.png"

    #get your images using glob
    imagelist = glob.glob(frame_path + "/*.png")

    #just take the even ones
    imagelist = sorted(imagelist)

    if len(imagelist) == 0:
        print "NO images"
        return

    img = Image.open(imagelist[0])
    size = img.size

    # Create the sprite
    width = 200;
    # print size
    size2 = (200, int(float(size[1])/size[0]*200.0))

    # Create size + middle frame + blank frame
    master_width = (width * 12)
    master_height = size2[1]

    # print  "  Master: ",(master_width, master_height)
    master = Image.new(
        mode='RGB',
        size=(master_width, master_height),
        color=(0,0,0))  # fully transparent and black

    sprite_images = [imagelist[i] for i in select_frames(len(imagelist))]

    for count, filename in enumerate(sprite_images):
        print "  %d. Adding: "%(count), " ", filename
        image = Image.open(filename)
        # os.unlink(filename)
        image2 = image.resize(size2, Image.ANTIALIAS)
        location = width*count
        master.paste(image2,(location,0))

    output = StringIO.StringIO()
    master.save(output, format='JPEG', quality=50)
    contents = output.getvalue()
    output.close()
    master.save(frame_path + "/sprite.jpg", format='JPEG', quality=70)
    # No need to update the clips
        # clips.update({"id" : clipid},{"$set" : { "middle_preview" : bson.Binary(contents), "preview_ready" : True}})
    print " Done .."

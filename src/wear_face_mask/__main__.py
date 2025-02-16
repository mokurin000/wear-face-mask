from wear_face_mask import WearFaceMask

if __name__ == "__main__":
    face_path = "imgs/test.jpg"
    save_path = "imgs/face_mask.jpg"

    wfm = WearFaceMask(torch_device="cuda")
    masked_face = wfm.wear_face_mask(face_path=face_path)
    masked_face.save(fp=save_path)

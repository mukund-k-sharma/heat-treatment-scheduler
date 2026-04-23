import qrcode

qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR Code (1 to 40)
    error_correction=qrcode.constants.ERROR_CORRECT_L, # Correction level
    box_size=10, # Pixels per box
    border=4,    # Minimum border thickness
)

qr.add_data("https://github.com/mukund-k-sharma/heat-treatment-scheduler")
qr.make(fit=True)

# Customize colors
img = qr.make_image(fill_color="black", back_color="white")
img.save("heat_treatment_scheduler_qr_code.png")
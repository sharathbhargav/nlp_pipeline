import PyPDF2 as ppdf
import pdfminer


f = open("/home/ullas/Downloads/pmse.pdf", 'rb')
pr = ppdf.PdfFileReader(f)
f1 = open("/home/ullas/abc.txt", 'w+')
for page in range(pr.numPages):
    pg = pr.getPage(page)
    s = pg.extractText()
    f1.write(s)


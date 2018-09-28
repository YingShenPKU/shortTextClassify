# When both values have been entered, check if the person
# is the right age to go on an 18-30 holiday (they must be over 18 and under 31).
# If they are, welcome them to the holiday, otherwise print a (polite) message refusing them entry.

from tkinter import * #imports the entire tkinter library
from tkinter import messagebox

root = Tk() #blank window called 'root'
root.title("基于命名实体识别的医学语义量化系统v1.0")
root.geometry("500x350+0+0") #size of root window

# heading = Label(root, text="Lets see if you are the right age!", font=("arial", 15, "bold"), fg="red").pack() #creates a heading of text
label_name = Label(root, text="医学描述:", font=("arial", 10, "bold")).place(x=50, y=50)
# label_age = Label(root, text="Please enter your age:", font=("arial", 20, "bold"), fg="green").place(x=10, y=200)


nameEntry_box = Entry(root, width=35, bg="white") #creates the name entry box
nameEntry_box.place(x=150, y=50) #defines the size of the entry box

# ageEntry_box =
# ageEntry_box.place(x=340, y=210)

Label(root,text="医学实体：", font=("arial", 10, "bold")).place(x=50, y=130)
Label(root, text="语义量化：", font=("arial", 10, "bold")).place(x=50, y=160)
Label(root, text="更多信息：", font=("arial", 10, "bold")).place(x=50, y=190)

from CRF.CNER import CNER
from classify.inputMatrix import classify
from ontologyInfo import ontologyInfo

# def print_item(event):
    # global listbox
    # print (listbox.get(listbox.curselection()))
def check_age(): #creates the function to be called once the button is pressed
    print('用户输入：', nameEntry_box.get())
    print('加载CNER模型...')
    diseases = CNER(nameEntry_box.get())
    # print(diseases)
    l1 = '，'.join(diseases)
    Label(root, text=l1, width=35, bg="white").place(x=150, y=130)
    print('加载HAC-LSTM模型...')
    classLabel = classify(nameEntry_box.get())
    classLabel1 = '轻微：' + str(classLabel['轻微']) + '，中等：' + str(classLabel['中等']) + '，严重：' + str(classLabel['严重'])
    Label(root, text=classLabel1, width=35, bg="white").place(x=150, y=160)

    diseases = list(diseases)
    var = StringVar()
    global listbox
    listbox = Listbox(root,width=35, height=5,bg="white",selectmode=BROWSE, listvariable = var)
    # listbox.bind('<ButtonRelease-1>', print_item)
    if len(diseases[0]) > 0:
        print('加载本体信息...')
        moreInfo = ontologyInfo(diseases[0])
        if len(moreInfo) > 0:
            print('检索到{%s}的本体信息'%diseases[0])
            for i in moreInfo:
                listbox.insert(END,i)
        else:
            print('没检索到{%s}的本体信息'%diseases[0])

    # scrl = Scrollbar(root)
    # # scrl.place(x=245, y=190)
    # scrl.pack(fill = Y)
    # listbox.configure(yscrollcommand=scrl.set)
    listbox.place(x=150, y=190)
    # scrl['command'] = listbox.yview
    # Label(root, text=info, width=35, height=5,bg="white",justify = 'left' ).place(x=150, y=190)

    # messagebox.showinfo(message=l1)


check = Button(root, text="请确认", width=10, height=1, bg="white", command= check_age).place(x=200, y=80) #creates the button and calls the function to be executed

root.mainloop() #mainloop creates an inf loop of the open window so it stays open
import requests
from bs4 import BeautifulSoup
import os
# import pprint

# # Playing with BeautifulSoup

# res = requests.get('https://news.ycombinator.com/')
# soup = BeautifulSoup(res.text, 'html.parser')

# print(res)
# print(res.text)

# print(soup.body)
# print(soup.body.contents)
# print(soup.find_all('a'))

# print(soup.title)
# print(soup.a)  # gives first a tag
# print(soup.find('a'))  # gives first a tag
# print(soup.find(id = 'score_26554065'))

# CSS Selector
# print(soup.select('a'))  # Selects all the a tags
# print(soup.select('.score'))  # Selects all the score classes. Use (.) for classes
# print(soup.select('#score_26554065')) # Selects all with specified id. Use (#) for ids

res = requests.get('https://www.google.com/search?q=donald+trump&sxsrf=ALeKk02_NfiBLRD7GqenHvntdpIEsnwhrA:1626269402465&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiXu6vQ1eLxAhXGzjgGHbzrCzYQ_AUoAnoECAEQBA&biw=1280&bih=648&dpr=1.5')
soup = BeautifulSoup(res.text,"html.parser")
links = []

images = soup.select('img[src^="data:image/"]')
for img in images:
    links.append(img['src'])
print(links)
# os.mkdir(r'D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\face_recognition\face_recognition_lib_method\dataset\donald_trump_web')

i=1
for index,img_link in enumerate(links):
    if i<=10:
        img_data = requests.get(img_link).content
        with open(r"D:\Projects\security_camera_face_recognition\Security_Camera_Face_Recognition\face_recognition\face_recognition_lib_method\dataset\donald_trump_web\\"+str(index+1)+'.jpg', 'wb+') as f:
            f.write(img_data)
        i+=1
    else:
        f.close()
        break

# res = requests.get('https://news.ycombinator.com/')
# soup = BeautifulSoup(res.text, 'html.parser')

# links = soup.select('.storylink')
# subtext = soup.select('.subtext')

# res2 = requests.get('https://news.ycombinator.com/news?p=2')
# soup2 = BeautifulSoup(res2.text, 'html.parser')

# links2 = soup2.select('.storylink')
# subtext2 = soup2.select('.subtext')

# mega_links, mega_subtext = links + links2, subtext+subtext2

# def sort_by_votes(hnlist):
#     return sorted(hnlist, key = lambda k:k['votes'], reverse=True)


# def create_custom_hn(links, subtext):
#     hn = []
#     for idx, item in enumerate(links):
#         title = links[idx].getText()
#         href = links[idx].get('href', None)
#         vote = subtext[idx].select('.score')
#         # print(vote)
#         if len(vote):
#             points = int(vote[0].getText().replace(' points', ''))
#             # print(int_votes)
#             if points >= 100:
#                 hn.append({'title': title, 'link':href, 'votes': points})
#     return sort_by_votes(hn)

# # print(create_custom_hn(links, subtext))
# # pprint.pprint(create_custom_hn(links, subtext))

# # For multiple pages
# pprint.pprint(create_custom_hn(mega_links, mega_subtext))
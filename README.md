## Requirements

- python==3.6
- torch==1.5

## How to run

Run the codes with the following commands on different datasets ("Movie & Book", "Movie & Music" and "Music & Book"). 

-->on Movie & Book dataset:
'''
!python p2fcdr.py --datasets movie_book_5_10_+book_movie_5_10_"
'''

-->on Music & Movie dataset:
'''
!python p2fcdr.py --datasets music_movie_5_10_+movie_music_5_10_"
'''

-->on Book & Music dataset:
'''
!python p2fcdr.py --datasets book_music_5_10_+music_book_5_10_"
'''

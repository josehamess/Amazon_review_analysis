from selenium import webdriver
import regex as re
import time


class ReviewExtract():
    def __init__(self, start_page_parts):
        self.start_page_parts = start_page_parts


    def review_finder(self, webpage_text):

        # finds the reviews in the scraped webpage #
        # returns a list of reviews #'

        text_body = re.findall(r'<span>(.*?)</span>', webpage_text)
        webpage_reviews = []
        counter = 0
        for text in text_body:
            if len(re.findall(r'"', text)) < 10:
                if text not in ['© 1996-2022, Amazon.com, Inc. or its affiliates', 'prime', 'Prime']:
                    if counter % 2 != 0:
                        webpage_reviews.append(text)
                    counter += 1

        return webpage_reviews
    

    def star_rating_finder(self, webpage_text):

        # finds the star rating of a review #
        # returns a list of star ratings #

        string_ratings = re.findall(r'<span class="a-icon-alt">(.*?).0 out of 5 stars</span></i></a>', webpage_text)

        return [int(x) for x in string_ratings]

    
    def next_page_link(self, webpage_parts, page_num):

        # creates the link for the next page of reviews based on the previous page link #
        # returns link as a string #
        webpage = webpage_parts[0] + webpage_parts[1] + str(page_num) + \
                    webpage_parts[2] + str(page_num)
        
        return webpage


    def scraper(self):

        # extracts all review text from review pages #
        # returns a long list of reviews #

        reviews_all = []
        star_ratings_all = []
        reviews = []
        star_ratings = []
        new_page = True
        page_num = 1
        driver = webdriver.Safari()
        while new_page == True:
            driver.get(self.next_page_link(self.start_page_parts, page_num))
            time.sleep(1)
            webpage_text = driver.page_source
            extracted_reviews = self.review_finder(webpage_text)
            extracted_ratings = self.star_rating_finder(webpage_text)
            if len(extracted_reviews) == len(extracted_ratings):
                reviews += extracted_reviews
                star_ratings += extracted_ratings
                if page_num % 50 == 0 and page_num > 0:
                    reviews_all += reviews
                    star_ratings_all += star_ratings
                    reviews = []
                    star_ratings = []
            if len(extracted_reviews) == 0:
                new_page = False
            page_num += 1
        driver.quit()

        return reviews_all, star_ratings_all
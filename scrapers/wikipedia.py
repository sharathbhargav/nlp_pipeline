# -*- coding: utf-8 -*-
import scrapy
from scrapy.selector import HtmlXPathSelector
from bs4 import BeautifulSoup


class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['https://en.wikipedia.org/wiki/Farm']

    def parse(self, response):
        hxp = HtmlXPathSelector(response)
        top_title = hxp.select("//h1[@class='firstHeading']/text()").extract()[0]
        text_plain = hxp.select(
            "//div[@id='bodyContent']/div[@id='mw-content-text']/div[@class='mw-parser-output']/p/text()").extract()
        text_bold = hxp.select(
            "//div[@id='bodyContent']/div[@id='mw-content-text']/div[@class='mw-parser-output']/p/b/text()").extract()
        text_href = hxp.select(
            "//div[@id='bodyContent']/div[@id='mw-content-text']/div[@class='mw-parser-output']/p/a/text()").extract()
        filename = top_title.replace(' ', '')
        floc = "/home/ullas/nltk_trial/corpora/op/" + filename + '.txt'
        f = open(floc, 'w+')
        f.write(top_title)
        f.write(' '.join(text_plain) + '\n')
        f.write(' '.join(text_bold) + '\n')
        f.write(' '.join(text_href) + '\n')
        f.close()


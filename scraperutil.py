from scrapy.crawler import CrawlerProcess
from scrapers.wikipedia import WikipediaSpider


process = CrawlerProcess()
process.crawl(WikipediaSpider)
process.start()

# __author__ = 'tixie'
import sys
sys.path.append(r"/home/wds/zhangjiong/Integration-model/seCrawler")

from scrapy.spiders import Spider
from ..common.searResultPages import searResultPages
from ..common.searchEngines import SearchEngineResultSelectors
from scrapy.selector import  Selector
import os

class keywordSpider(Spider):
    name = 'keywordSpider'
    allowed_domains = ['bing.com','google.com','baidu.com']
    start_urls = []
    searchEngine = None
    selector = None
    script_dir = os.path.dirname(__file__)


    def __init__(self, keyword_file, se = 'bing', pages = 50,  *args, **kwargs):
        super(keywordSpider, self).__init__(*args, **kwargs)
        self.searchEngine = se.lower()
        self.selector = SearchEngineResultSelectors[self.searchEngine]
        abs_file_path = os.path.abspath(os.path.join(self.script_dir, "..", "..", keyword_file))
        print(abs_file_path)
        with open(abs_file_path) as f:
            for keyword in f:
                pageUrls = searResultPages(keyword, se, int(pages))
                for url in pageUrls:
                    print(url)
                    self.start_urls.append(url)

    def parse(self, response):
        # items = Selector(response).xpath(self.selector)
        # with open(os.path.abspath(os.path.join(self.script_dir, "..", "..", 'response.html')), 'w') as f:
        #     f.write(response.text)
        #     f.close()
        items = Selector(response).xpath('//ol[@id="b_results"]/li[@class="b_algo"]')
        for index, item in enumerate(items):
            yield {
                'title': ''.join(item.xpath('./div[@class="b_title"]/h2/a//text()').extract()),
                'url': item.xpath('./div[@class="b_title"]/h2/a/@href').extract()[0],
                'content': ''.join(item.xpath('./div[@class="b_caption"]/p//text()').extract())
            }

        pass

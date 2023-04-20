# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.exceptions import DropItem
import  logging

class SespiderPipeline(object):
    def __init__(self):
        self.file = open('spider_text.txt', 'w+')
        #self.bloomFilter = rBloomFilter.rBloomFilter(100000, 0.01, 'bing')

    def process_item(self, item, spider):
        # print(item['url']+'\n')
        self.file.write(item['url']+'\t'+item['title']+'\t'+item['content']+'\n')
        print('title: ', item['title'], sep='\t')
        print('content: ', item['content'], sep='\t')
        return item

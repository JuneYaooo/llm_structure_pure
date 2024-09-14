from flask import Flask, request, jsonify, make_response
from infer_llama3 import LLMPredict
import time
import pandas as pd
import os
import logging
import datetime
from config.common_config import *

app = Flask(__name__)
fd_pred = LLMPredict()
os.makedirs('log', exist_ok=True)
logging.basicConfig(level=logging.INFO, filename='log/execute.log', filemode='a')
logger = logging.getLogger(__name__)
logger.info('finished init model!')

@app.route('/data', methods=['GET', 'POST'])
def get_data():
    
    logging.basicConfig(level=logging.INFO, filename='log/execute.log', filemode='a')
    logger = logging.getLogger(__name__)
    if request.method not in ['GET', 'POST']:
        logger.info('Method not allowed')
        return make_response(jsonify({'err': 405, 'msg': 'Method not allowed'}), 405)

    if not request.is_json:
        logger.info('Invalid JSON data')
        return make_response(jsonify({'err': 400, 'msg': 'Invalid JSON data'}), 400)

    data_dict = request.get_json()
    
    start_time = time.time()
    if isinstance(data_dict, list):
        # 如果是列表，则批处理数据
        fnumber = len(data_dict)
        for data_item in data_dict:
            if not all(key in data_item  for key in ['project', 'field_en', 'field','raw_text']):
                logger.info('Missing required fields')
                return make_response(jsonify({'err': 400, 'msg': 'Missing required fields'}), 400)
            medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=data_item['project'])
            if data_item['field_en'] not in medical_logic['字段英文名称'].values:
                logger.info('Please confirm if the field field_name is the correct English field name')
                return make_response(jsonify({'err': 400, 'msg': 'Please confirm if the field field_name is the correct English field name'}), 400)
        
        try:
            res,token_num = fd_pred.get_batch_result(data_dict)
        except Exception as e:
            print('e',e)
            print('发生错误，强制退出程序！')
            logger.info('多字段，发生错误，强制退出程序！',e)
            start_time_format = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
            store_data = {'err': 1, 'msg': 'error', 'data_dict': data_dict, 'error_msg': e, 'startTime': start_time_format}
            log_path = 'log/error.log'
            with open(log_path, 'a', encoding='utf-8') as file:
                file.write(str(store_data) + '\n')
            # exit()
            return make_response(jsonify({'err': 500, 'msg': 'Failed to get result from Model Predict!'}), 500)

    elif isinstance(data_dict, dict):
        # 如果是字典，则单字段处理
        fnumber = 1
        if not all(key in data_dict for key in ['project', 'field_en', 'field','raw_text']):
            
            logger.info('Missing required fields')
            return make_response(jsonify({'err': 400, 'msg': 'Missing required fields'}), 400)
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=data_dict['project'])
        if data_dict['field_en'] not in medical_logic['字段英文名称'].values:
            logger.info('Please confirm if the field field_name is the correct English field name')
            return make_response(jsonify({'err': 400, 'msg': 'Please confirm if the field field_name is the correct English field name'}), 400)
        
        try:   
            res,token_num = fd_pred.get_result(data_dict["project"],data_dict["field_en"],data_dict["field"],data_dict["raw_text"])
        except Exception as e:
            print('单字段，发生错误！')
            logger.info('单字段，发生错误！',e)
            start_time_format = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
            store_data = {'err': 1, 'msg': 'error', 'data_dict': data_dict, 'error_msg': e, 'startTime': start_time_format}
            log_path = 'log/error.log'
            with open(log_path, 'a', encoding='utf-8') as file:
                file.write(str(store_data) + '\n')
            return make_response(jsonify({'err': 500, 'msg': 'Failed to get result from Model Predict!'}), 500)

    else:
        # 其他类型则抛出异常
        logger.info('Unsupported data type, please use list or dict')
        return make_response(jsonify({'err': 400, 'msg': 'Unsupported data type, please use list or dict'}), 500)
    end_time = time.time()
    cost_time = end_time - start_time
    print('====cost_time===',cost_time)
    start_time_format = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    store_data = {'err': 0, 'msg': 'Success', 'predResult': res, 'costTime': cost_time, 'startTime': start_time_format, 'fnumber':fnumber, 'token_num': token_num}
    log_path = 'log/time.log'
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write(str(store_data) + '\n')

    return make_response(jsonify({'err': 0, 'msg': 'Success', 'predResult': res, 'costTime': cost_time}), 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=46000)

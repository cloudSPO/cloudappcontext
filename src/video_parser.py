import pandas as pd
import numpy as np
import os
import glob
import math
import sys
from config import bazne_cijene_video

class Video_Parser:
    def __init__(self, path, mode):        
        self.extension = '.csv'
        self.root_folder = path
        self.mode = mode
        self.metrics =  {
            'cpu_util': ['|', 2, 'Avg. CPU utilization (%)'],
            'diskreadbytes': ['|', 2, 'Disk read bytes'],
            'diskwritebytesrate': ['|', 2, 'Disk write bytes rate'],
            'memory_unused': [' ', 2, 'Avg RAM (unused, KiB)'],
            'network_incoming_bytes': ['|', 2, 'NW Incoming bytes'],
            'network_incoming_bytes_rate': ['|', 2, 'NW Incoming byte rate'],
            'network_outgoing_bytes': ['|', 2, 'NW outgoing bytes'],
            'network_outgoing_bytes_rate': ['|', 2,'NW outgoing byte rate']    
        }
                
        self.columns_order_cost = ['Users', 
                        'playlist', 
                        'chunk', 
                        'End-to-end',
                        'No. SLO violations, threshold 5s (%)',
                        'Avg. CPU utilization (%)',
                        'Max CPU utilization (%)',
                        'NW Incoming byte rate',
                        'NW Incoming bytes',
                        'NW Ingress (GB)',
                        'NW outgoing byte rate',
                        'NW outgoing bytes',
                        'NW Egress (GB)',
                        'Avg RAM (unused, KiB)',
                        'Avg RAM used (MB)',
                        'Min free RAM (KiB)',
                        'Max used RAM (MB)',
                        'Disk write bytes rate',
                        'Disk read bytes',
                        'Used storage(image size, GB)',
                        'Min RAM',
                        'Min vCPU',
                        'Century Link',
                        'Google',
                        'Azure',
                        'Amazon',
                        'Oracle',
                        'Alibaba',
                        'DigitalOcean']
                
        self.columns_order_qos = ['Users', 
                        'No. SLO violations, threshold 5s (%)',
                        'Avg. CPU utilization (%)',
                        'Max CPU utilization (%)',
                        'NW Incoming byte rate',
                        'NW Incoming bytes',
                        'NW Ingress (GB)',
                        'NW outgoing byte rate',
                        'NW outgoing bytes',
                        'NW Egress (GB)',
                        'Avg RAM (unused, KiB)',
                        'Avg RAM used (MB)',
                        'Min free RAM (KiB)',
                        'Max used RAM (MB)',
                        'Disk write bytes rate',
                        'Disk read bytes',
                        'Used storage(image size, GB)',
                        'Min RAM',
                        'Min Storage (GB)',
                        'Min vCPU']
        
        self.sheet_names = {'XLargeDB_smallinstance':'openmrs_small_database_xlarge', 
                            'XLargeDB_mediuminstance':'openmrs_medium_database_xlarge', 
                            'XLargeDB_largeinstance':'openmrs_large_database_xlarge',
                            'LargeDB_smallinstance':'openmrs_small_database_large', 
                            'LargeDB_mediuminstance':'openmrs_medium_database_large', 
                            'LargeDB_largeinstance':'openmrs_large_database_large'}
    
    def parse_single(self, path_to_folder, user, sheet_name):
        if self.mode == 'cost':
            return self.parse_single_cost(path_to_folder, user, sheet_name)
        if self.mode == 'qos':
            return self.parse_single_qos(path_to_folder, user, sheet_name)
        
    def parse_single_qos(self, path_to_folder, user, sheet_name):  
         # general info
        ginfo = {}
        # metrics analysis results
        metrics_avg = pd.DataFrame()
        
        with open(path_to_folder+'general_info.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                fields = line.split(':')
                ginfo[fields[0].strip()] = "".join(fields[1:]).strip().split(" ")[0].strip()


         # general metrics (defined in 'metrics' dictionary)
        for metric_name, opts in self.metrics.items():
            data = pd.read_csv(path_to_folder + metric_name + self.extension, sep=opts[0], header=None, decimal=",", error_bad_lines=False)
            ind = int(opts[1])
            metrics_avg[opts[2]] = data[ind]
        
        # max cpu util
        data = pd.read_csv(path_to_folder + 'cpu_util' + self.extension, sep='|', header=None, decimal=",", error_bad_lines=False)
        metrics_avg['Max CPU utilization (%)'] = data[2]
        length = len(data[2])

        # ingress & egress
        metrics_avg['NW Ingress (GB)'] = metrics_avg['NW Incoming bytes']/(1024**3)
        metrics_avg['NW Egress (GB)'] = metrics_avg['NW outgoing bytes']/(1024**3)
        metrics_avg['NW Ingress (GB)'] *= (4*24*30)
        metrics_avg['NW Egress (GB)'] *= (4*24*30)
        
        # ram usage metrics
        ram_capacity =  pd.Series([float(ginfo['Available RAM']) for i in range(length)])
        metrics_avg['Used storage(image size, GB)'] = pd.Series([float(ginfo['HDD']) for i in range(length)])
        metrics_avg['Min vCPU'] = pd.Series([float(ginfo['Vcpus']) for i in range(length)])
        metrics_avg['Min RAM'] = pd.Series([float(float(ginfo['RAM'].strip())/1024.0) for i in range(length)])
        metrics_avg['Users'] = pd.Series([user for i in range(length)])

        # memory data
        data = pd.read_csv(path_to_folder + 'memory_unused' + self.extension, sep=' ', header=None, decimal=",", error_bad_lines=False)
        step = math.ceil(data[2].size/length)
        start = 0
        end = step
        data_range = pd.Series()
        avg_ram_unused = []
        min_ram_unused = []
        avg_ram_used = []
        max_ram_used = []

        for i in range(0, data[2].size, step):
            data_range = data[2].iloc[start:end]
            start = end
            end += step
            avg_ram_unused.append(data_range.mean())
            min_ram_unused.append(data_range.min())
            avg_ram_used.append((data_range - ram_capacity[0]).abs().mean() / 976.563)
            max_ram_used.append((data_range - ram_capacity[0]).abs().max() / 976.563)
    
        metrics_avg['Avg RAM (unused, KiB)'] = pd.Series(avg_ram_unused)
        metrics_avg['Min free RAM (KiB)'] = pd.Series(min_ram_unused)
        metrics_avg['Avg RAM used (MB)'] = pd.Series(avg_ram_used)
        metrics_avg['Max used RAM (MB)'] = pd.Series(max_ram_used)
        
        # SLo violations data        
        jMeter_data = None
        for filename in os.listdir(path_to_folder):
            root, ext = os.path.splitext(filename)
            if root.startswith('jMeter') and not root.endswith('transaction') and ext == '.csv':
                jMeter_data = pd.read_csv(path_to_folder + filename, sep=',', decimal=",", error_bad_lines=True)
                # Http Transaction Controller
                stream = jMeter_data[jMeter_data['label'] == 'stream']
                end_to_end = stream['elapsed']
                
                slo_length = end_to_end.shape[0]
                step = math.ceil(slo_length/length)
                start = 0
                end = step
                data_range = pd.Series()
                slo_violations = []
                end2end = []

                for i in range(0, slo_length, step):
                    data_range = end_to_end.iloc[start:end]
                    start = end
                    end += step                           

                    #slo_viol
                    violated = sum(i > 5000 for i in data_range)
                    transaction_len = step
                    slo_violations.append(round((float(violated)/float(transaction_len)*100), 2))
                    end2end.append(data_range.mean())

                metrics_avg['No. SLO violations, threshold 5s (%)'] = pd.Series(slo_violations)
                metrics_avg['End-to-end'] = pd.Series(end2end)
                break
        
        #prepare cost info for calculating prices
        cost_info = {}
        cost_info['metrics'] = {'Min vCPU': metrics_avg['Min vCPU'], 
                                'Min RAM': metrics_avg['Min RAM'], 
                                'Used storage(image size, GB)': metrics_avg['Used storage(image size, GB)'], 
                                'NW Egress (GB)': metrics_avg['NW Egress (GB)']}
        
        #save calculated prices to metric_avg
        prices_dict = self.calculate_price_qos(cost_info, sheet_name)
        for provider in prices_dict:
            metrics_avg[provider] = prices_dict[provider]
        
        metrics_avg.sort_values(by='Users', inplace=True)
        return metrics_avg
        
    def parse_single_cost(self, path_to_folder, user, sheet_name):
        # general info
        ginfo = {}
        # metrics analysis results
        metrics_avg = {}
        
        # 'filename': ['separator', column_index]
        with open(path_to_folder+'general_info.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                fields = line.split(':')
                ginfo[fields[0].strip()] = "".join(fields[1:]).strip().split(" ")[0].strip()

        # general metrics (defined in 'metrics' dictionary)
        for metric_name, opts in self.metrics.items():
            data = pd.read_csv(path_to_folder + metric_name + self.extension, sep=opts[0], header=None, decimal=",", error_bad_lines=False)
            ind = int(opts[1])
            metrics_avg[opts[2]] = data[ind].mean()

        # max cpu util    
        data = pd.read_csv(path_to_folder + 'cpu_util' + self.extension, sep='|', header=None, decimal=",", error_bad_lines=False)
        metrics_avg['Max CPU utilization (%)'] = data[2].max()
        
        # ingress & egress
        metrics_avg['NW Ingress (GB)'] = metrics_avg['NW Incoming bytes']/(1024**3)
        metrics_avg['NW Egress (GB)'] = metrics_avg['NW outgoing bytes']/(1024**3)
        metrics_avg['NW Ingress (GB)'] *= (4*24*30)
        metrics_avg['NW Egress (GB)'] *= (4*24*30)
        
        # ram usage metrics
        ram_capacity = float(ginfo['Available RAM'])

        data = pd.read_csv(path_to_folder + 'memory_unused' + self.extension, sep=' ', header=None, decimal=",", error_bad_lines=False)
        metrics_avg['Avg RAM (unused, KiB)'] = data[2].mean()
        metrics_avg['Min free RAM (KiB)'] = data[2].min()
        metrics_avg['Avg RAM used (MB)'] = (data[2] - ram_capacity).abs().mean() / 976.563
        metrics_avg['Max used RAM (MB)'] = (data[2] - ram_capacity).abs().max() / 976.563
        metrics_avg['Used storage(image size, GB)'] = float(ginfo['HDD'])
        metrics_avg['Min RAM'] = float(float(ginfo['RAM'].strip())/1024.0)
        metrics_avg['Min vCPU'] = float(ginfo['Vcpus'])
        
        #prepare cost info for calculating prices
        cost_info = {}
        cost_info['metrics'] = {'Min vCPU': metrics_avg['Min vCPU'], 
                                'Min RAM': metrics_avg['Min RAM'], 
                                'Used storage(image size, GB)': metrics_avg['Used storage(image size, GB)'], 
                                'NW Egress (GB)': metrics_avg['NW Egress (GB)']}
        
        #save calculated prices to metric_avg
        prices_dict = self.calculate_price_cost(cost_info, sheet_name)
        for provider in prices_dict:
            metrics_avg[provider] = prices_dict[provider]
        
        #reponse time metrics
        jMeter_data = None
        for filename in os.listdir(path_to_folder):
            root, ext = os.path.splitext(filename)
            if root.startswith('jMeter') and not root.endswith('transaction') and ext == '.csv':
                jMeter_data = pd.read_csv(path_to_folder + filename, sep=',', decimal=",", error_bad_lines=False)

                # Http Request1
                httpReq1 = jMeter_data[jMeter_data['label'] == 'playlist']
                # Http Request2
                httpReq2 = jMeter_data[jMeter_data['label'] == 'chunk']
                # Http Request3
                httpReq3 = jMeter_data[jMeter_data['label'] == 'stream']
                
                metrics_avg['playlist'] = httpReq1['elapsed'].mean()
                metrics_avg['chunk'] = httpReq2['elapsed'].mean()
                metrics_avg['End-to-end'] = httpReq3['elapsed'].mean()
                #slo_viol
                violated = sum(i > 5000 for i in httpReq3['elapsed'])
                transaction_len = len(httpReq3['elapsed'])
                metrics_avg['No. SLO violations, threshold 5s (%)'] = round((float(violated)/float(transaction_len)*100), 2)
                metrics_avg['Users'] = int(user)
        res = pd.DataFrame.from_records([metrics_avg])            
        return res[self.columns_order_cost]
    
    def calculate_price_qos(self, cost_info, input_sheet_name):
        #read price file
        excel_file = bazne_cijene_video
        price_xlsx = pd.ExcelFile(excel_file)
        sheet_names = price_xlsx.sheet_names

        #save cost for every instance-database combination
        cost_dict = {}
        for sheet_name in sheet_names:
            single_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
            cost_dict[sheet_name] = single_sheet

        #read metrics
        egress = cost_info['metrics']['NW Egress (GB)']
        storage = cost_info['metrics']['Used storage(image size, GB)']
        minram = cost_info['metrics']['Min RAM']
        vcpu = cost_info['metrics']['Min vCPU']
        length = len(egress)
        
        providers = ['Century Link', 'Google', 'Azure', 'Amazon', 'Oracle', 'Alibaba', 'DigitalOcean']
        #init return cost
        return_cost = {}
        for provider in providers:
            return_cost[provider] = []
            for index in range(length):
                return_cost[provider] = [0]*length

        #init provider cost
        provider_cost = {}
        for provider in providers:
            provider_cost[provider] = [0]*length

        #read provider base price
        cost = cost_dict[input_sheet_name]
        for index, row in cost.iterrows():
            for i in range(length):
                if row['Min vCPU'] == cost_info['metrics']['Min vCPU'][i] and row['Min RAM'] == cost_info['metrics']['Min RAM'][i] and row['Used storage(image size, GB)'] <= cost_info['metrics']['Used storage(image size, GB)'][i]:
                    for provider in providers:
                        provider_cost[provider][i] = row[provider]

        #cost per egress from sheet
        egress_cost = cost_dict['egress_cost']
        
        ##special cases
        #Azure
        all_cost = provider_cost['Azure']
        for index in range(0, len(egress)):
            e = egress[index]
            c = all_cost[index]
            if e < 5:
                pass
            elif 5<e<10*1024:
                c = c + e*0.087
            elif 10*1024<e<50*1024:
                c = c + e*0.083
            elif 50*1024<e<150*1024:
                c = c + e*0.07
            elif 150*1024<e<500*1024:
                c = c + e*0.05
            return_cost['Azure'][index]= c
        #Oracle
        all_cost = provider_cost['Oracle']
        for index in range(0, len(egress)):
            e = egress[index]
            c = all_cost[index]
            if e < 10*1024:
                pass
            else:
                c = c + e*0.0085
            return_cost['Oracle'][index]= c
        
        #non-special cases
        #go through providers in sheet, add price according to egress
        for provider in egress_cost:
            for index in range(0, len(egress)):
                e = egress[index]
                c = provider_cost[provider][index]
                for i, row in egress_cost.iterrows():
                    c = c + e*row[provider]
                return_cost[provider][index] = c

        return return_cost
    
    def calculate_price_cost(self, cost_info, input_sheet_name):
        #read price file
        excel_file = bazne_cijene_video
        price_xlsx = pd.ExcelFile(excel_file)
        sheet_names = price_xlsx.sheet_names
        
        #save cost for every instance-database combination
        cost_dict = {}
        for sheet_name in sheet_names:
            single_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=0)
            cost_dict[sheet_name] = single_sheet

        #read metrics
        egress = cost_info['metrics']['NW Egress (GB)']
        storage = cost_info['metrics']['Used storage(image size, GB)']
        minram = cost_info['metrics']['Min RAM']
        vcpu = cost_info['metrics']['Min vCPU']
        
        providers = ['Century Link', 'Google', 'Azure', 'Amazon', 'Oracle', 'Alibaba', 'DigitalOcean']
        #init return cost
        return_cost = {}
       
        #init provider cost
        provider_cost = {}
        for provider in providers:
            provider_cost[provider] = 0

        #read provider base price
        cost = cost_dict[input_sheet_name]
        for index, row in cost.iterrows():
            if row['Min vCPU'] == cost_info['metrics']['Min vCPU'] and row['Min RAM'] == cost_info['metrics']['Min RAM'] and row['Used storage(image size, GB)'] <= cost_info['metrics']['Used storage(image size, GB)']:
                for provider in providers:
                    provider_cost[provider] = row[provider]
        
        ##special cases
        #Azure
        all_cost = provider_cost['Azure']
        e = egress
        c = all_cost
        if e < 5:
            pass
        elif 5<e<10*1024:
            c = c + e*0.087
        elif 10*1024<e<50*1024:
            c = c + e*0.083
        elif 50*1024<e<150*1024:
            c = c + e*0.07
        elif 150*1024<e<500*1024:
            c = c + e*0.05
        return_cost['Azure'] = c
        #Oracle
        all_cost = provider_cost['Oracle']
        e = egress
        c = all_cost
        if e < 10*1024:
            pass
        else:
            c = c + e*0.0085
        return_cost['Oracle'] = c
        
        #cost per egress from sheet
        egress_cost = cost_dict['egress_cost']
        #non-special cases
        #go through providers in sheet, add price according to egress
        for provider in egress_cost:
            e = egress
            c = provider_cost[provider]
            for i, row in egress_cost.iterrows():
                c = c + e*row[provider]
            return_cost[provider] = c
        
        return return_cost
    
    def parse(self): 
        data = {}
        for mrs_instance in sorted(os.listdir(self.root_folder)):
            tmp = []
            sheet_name = mrs_instance

            for mrs_instance_users in sorted(os.listdir(self.root_folder+"/"+mrs_instance)):
                path_to_folder = self.root_folder+"/"+mrs_instance+"/"+mrs_instance_users+"/"
                single_metrics_result = self.parse_single(path_to_folder, mrs_instance_users.split('user')[0], sheet_name)
                tmp.append(single_metrics_result)

            tmp2 = pd.concat(tmp, sort=True)
            tmp2.sort_values(by='Users', inplace=True)
            tmp2.reset_index(drop=True, inplace=True)

            if self.mode == 'cost':
                tmp2 = tmp2[self.columns_order_cost]
            data[sheet_name] = tmp2
        return data
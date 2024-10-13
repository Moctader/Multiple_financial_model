# pip install redis
import redis, json
import json

class create_redis_instance:
    def __init__(self):
        self.instance = redis.Redis(host='193.166.180.240', port=6379, db=0)

    def __del__(self):
        print('REDIS INSTANCE DIED')
    
    ########################################################################################################
    ########################################################################################################
    
    def set(self, key: str, value: str|int|float|dict):
        assert type(key) == str, '[REDIS ERROR] THE KEY MUST BE A STRING'
        assert isinstance(value, (str, int, float, dict)), '[REDIS ERROR] VALUE TYPE MUST BE STR|INT|FLOAT|DICT'
        temp_value = value

        # STRINGIFY DICTS
        if type(value) == dict:
            temp_value = json.dumps(temp_value)
        
        result = self.instance.set(key, temp_value)
        assert result == 1, f"[REDIS ERROR] SETTING KEY '{key}' FAILED"

    ########################################################################################################
    ########################################################################################################

    def parse_value(self, raw_value):
        try:
            # DECODE IT
            stringified_value = raw_value.decode('utf-8')

            # IS IT AN INTEGER?
            if stringified_value.isdigit():
                return int(stringified_value)

            # IS IT A FLOAT?
            try:
                return float(stringified_value)
            except:
                pass

            # IS IT JSON?
            try:
                return json.loads(stringified_value)
            except:
                pass
            
            # OTHERWISE, ITS A STRING
            return stringified_value
        except Exception as error:
            print(f'[REDIS PARSING ERROR] {error}')

    ########################################################################################################
    ########################################################################################################
    
    def get(self, key: str):
        assert type(key) == str, '[REDIS ERROR] THE KEY MUST BE A STRING'

        # MAKE SURE THE VALUE EXISTS
        raw_value = self.instance.get(key)
        assert raw_value != None, f"[REDIS ERROR] KEY '{key}' HAS NO VALUE"
        
        # DECODE & PARSE IT
        return self.parse_value(raw_value)

    ########################################################################################################
    ########################################################################################################
    
    def exists(self, key: str):
        assert type(key) == str, '[REDIS ERROR] THE KEY MUST BE A STRING'
        return True if self.instance.exists(key) else False
    
    ########################################################################################################
    ########################################################################################################
    
    # MAKE SURE THE KEY EXISTS, THEN TRY TO DELETE IT
    def delete(self, key: str):
        assert type(key) == str, '[REDIS ERROR] THE KEY MUST BE A STRING'
        
        assert self.exists(key), f"[REDIS ERROR] KEY '{key}' DOES NOT EXIST"
        assert self.instance.delete(key) == 1, f"[REDIS ERROR] KEY '{key}' DELETION FAILED"
        
########################################################################################################
########################################################################################################

from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import Consumer, TopicPartition

# LOAD THE GLOBAL CONFIG & STITCH TOGETHER THE KAFKA CONNECTION STRING
kafka_brokers = '193.166.180.240:11001'

class create_admin_client:
    def __init__(self):

        # ATTEMPT TO CONNECT TO THE CLUSTER
        self.instance = AdminClient({
            'bootstrap.servers': kafka_brokers,
        })

        self.check_connection()

    ########################################################################################################
    ########################################################################################################

    # MAKE SURE KAFKA CONNECTION IS OK
    def check_connection(self):
        try:
            metadata = self.instance.list_topics(timeout=2)
            return True
        except Exception as error:
            raise Exception(f'[KAFKA CONNECTION ERROR] {error}') 

    ########################################################################################################
    ########################################################################################################

    def summarize_consumer_groups(self):
        group_names = []
        container = []

        # KAFKA CONSUMER GROUP STATES
        # https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#confluent_kafka.ConsumerGroupState
        kafka_states = {
            0: 'UNKNOWN',
            1: 'PREPARING_REBALANCE',
            2: 'COMPLETING_REBALANCE',
            3: 'STABLE',
            4: 'DEAD',
            5: 'EMPTY',
        }

        # QUERY ALL GROUP NAMES
        for item in self.instance.list_consumer_groups().result().valid:
            group_names.append(item.group_id)

        # QUERY GROUP INFORMATION
        for key, promise in self.instance.describe_consumer_groups(group_names).items():
            dict_data = promise.result().__dict__

            # UPDATE MEMBERS KEY TO COUNT INSTEAD
            dict_data['members'] = len(dict_data['members'])

            # PARSE ERROR INTS TO VERBOSE ALTERNATIVE
            try:
                state_id = int(dict_data['state'].__dict__['_value_'])
                dict_data['state'] = kafka_states[state_id]
            except:
                dict_data['state'] = 'FALLBACK'

            # GET RID OF USELESS GARBAGE
            del dict_data['coordinator']

            container.append(dict_data)

        return container

    ########################################################################################################
    ########################################################################################################

    def summarize_topics(self):
        try:
            formatted_topics = {}
            
            # CREATE A TEMP CONSUMBER TO READ TOPIC OFFSETS
            temp_consumer = Consumer({
                'bootstrap.servers': kafka_brokers,
                'group.id': 'offset_checker_group',
                'auto.offset.reset': 'earliest'
            })

            # PARSE THROUGH TOPIC DETAILS
            for topic_name, topic_metadata in self.instance.list_topics().topics.items():

                # SKIP THE OFFSETS TOPIC
                if global_config.backend.hide_auxillary and topic_name == '__consumer_offsets':
                    continue

                # NAME & THE NUMBER OF PARTITIONS
                formatted_topics[topic_name] = {
                    'num_partitions': len(topic_metadata.partitions),
                    'offsets': {}
                }

                # PARTITION OFFSETS
                for partition_id, _ in topic_metadata.partitions.items():
                    tp = TopicPartition(topic_name, partition_id)
                    earliest, latest = temp_consumer.get_watermark_offsets(tp, timeout=10)
                    # container[topic_name]['offsets'][partition_id] = tp.offset

                    formatted_topics[topic_name]['offsets'][partition_id] = {
                        'earliest': earliest,
                        'latest': latest
                    }
            
            return formatted_topics
        
        except Exception as error:
            raise Exception(f'[TOPIC SUMMARY ERROR] {error}') 
    
    ########################################################################################################
    ########################################################################################################

    def topic_exists(self, target_topic):
        try:

            # FISH OUT ALL EXISTING TOPIC NAMES
            existing_topics = [name for name, _ in self.instance.list_topics().topics.items()]

            # RETURN TRUE FOR DUPLICATES, OTHERWISE FALSE
            for topic in existing_topics:
                if topic == target_topic:
                    return True
            
            return False
        
        except Exception as error:
            raise Exception(f'[TOPIC EXISTS ERROR] {error}') 
    
    ########################################################################################################
    ########################################################################################################
    
    # ATTEMPT TO CREATE A NEW TOPIC
    def create_topic(self, name, num_partitions):
        try:

            # THROW ERROR IF TOPIC ALREADY EXISTS
            if self.topic_exists(name):
                raise Exception('TOPIC ALREADY EXISTS')

            # OTHERWISE, CREATE IT
            self.instance.create_topics(
                new_topics=[NewTopic(
                    topic=name,
                    num_partitions=num_partitions,
                    replication_factor=1,
                )]
            )

        except Exception as error:
            raise Exception(f'[CREATE TOPIC ERROR] {error}') 

########################################################################################################
########################################################################################################

from cassandra.cluster import Cluster
cassandra_brokers = [('193.166.180.240', 12001)]

class create_cassandra_instance:
    def __init__(self, HIDE_LOGS=False):
        cluster = Cluster(cassandra_brokers)
        self.instance = cluster.connect()

        if HIDE_LOGS:
            global VERBOSE
            VERBOSE = False

    def __del__(self):
        self.instance.shutdown()

    ########################################################################################################
    ########################################################################################################

    # FREELY EXECUTE ANY CQL QUERY
    def query(self, query):
        try:
            return self.instance.execute(query)
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA FREE-QUERY ERROR] => {parsed_error}')

    ########################################################################################################
    ########################################################################################################
    
    # CASSANDRA DRIVER ERRORS ARE VERY MESSY
    # THIS ATTEMPT TO PARSE THEM TO MAKE EVERYTHING MORE HUMAN-READABLE
    def parse_error(self, error):
        stringified_error = str(error)
        
        # TRY TO REGEX MATCH THE ERROR PATTERN
        match = re.search(r'message="(.+)"', stringified_error)

        # MATCH FOUND, RETURN ISOLATED ERROR MSG
        if match:
            return match.group(1)
        
        # OTHERWISE, RETURN THE WHOLE THING
        return stringified_error
    
    ########################################################################################################
    ########################################################################################################

    # READ DATA FROM THE DATABASE
    def read(self, query: str, sort_by=False) -> list[dict]:
        try:
            container = []

            # PERFORM THE TABLE QUERY
            query_result = self.instance.execute(query)

            # PARSE EACH ROW AS A DICT
            for item in query_result:
                container.append(item._asdict())

            if VERBOSE: misc.log(f'[CASSANDRA] READ {len(container)} FROM DATABASE')
            
            # SORT BY KEY WHEN REQUESTED
            if sort_by:
                return sorted(container, key=lambda x: x[sort_by])

            # OTHERWISE, RETURN UNSORTED
            return container
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA READ ERROR] => {parsed_error}')
    
    ########################################################################################################
    ########################################################################################################

    # COUNT TABLE ROWS -- SELECT COUNT(*) FROM ...
    def count(self, count_query: str) -> int:
        return int(self.read(count_query)[0]['count'])

    ########################################################################################################
    ########################################################################################################

    # FULL DATABASE OVERVIEW (KEYSPACES)
    def write(self, keyspace_table: str, row: dict):
        try:

            # SPLIT THE KEYS & VALUES
            columns = list(row.keys())
            values = list(row.values())

            # STITCH TOGETHER THE QUERY STRING
            query_string = f'INSERT INTO {keyspace_table} ('
            query_string += ', '.join(columns)
            query_string += ') values ('
            query_string += ', '.join(['?'] * len(columns))
            query_string += ');'
            
            # CONSTRUCT A PREPARED STATEMENT & EXECUTE THE DB WRITE
            prepared_statement = self.instance.prepare(query_string)
            self.instance.execute(prepared_statement, values)

            if VERBOSE: misc.log('[CASSANDRA] WROTE TO DATABASE')
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA WRITE ERROR] => {parsed_error}')
        
    ########################################################################################################
    ########################################################################################################

    # CREATE A NEW KEYSPACE
    def create_keyspace(self, keyspace_name):
        try:
            self.instance.execute("""
                CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {
                    'class': 'SimpleStrategy', 
                    'replication_factor': '1'
                };
            """ % keyspace_name).all()

        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA KEYSPACE ERROR] {parsed_error}')

    ########################################################################################################
    ########################################################################################################

    def create_table(self, keyspace_name, table_name, columns, indexing):
        try:

            # MAKE SURE PRIMARY KEYS ARE OK
            for key in indexing:
                col_list = list(columns.keys())
                
                if key not in col_list:
                    raise Exception(f"PRIMARY KEY '{key}' IS NOT A VALID COLUMN")
            
            # CREATE KEYSPACE IF NECESSARY
            self.create_keyspace(keyspace_name)

            # BASE QUERY
            query = f'CREATE TABLE {keyspace_name}.{table_name} ('
            
            # LOOP IN COLUMNS
            for column_name, column_type in columns.items():
                query += f'{column_name} {column_type}, '
                
            # ADD PRIMARY KEYS
            key_string = ', '.join(indexing)
            query += f'PRIMARY KEY({key_string}));'

            # CREATE THE TABLE
            self.instance.execute(query)

        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA CREATE ERROR] {parsed_error}')

    ########################################################################################################
    ########################################################################################################
 
    def drop_table(self, keyspace_name: str, table_name: str):
        try:
            self.instance.execute(f'DROP TABLE IF EXISTS {keyspace_name}.{table_name}')

        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA DROP ERROR] {parsed_error}')

########################################################################################################
########################################################################################################

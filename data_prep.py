import pandas as pd
from pathlib import Path

# disable truncated output
pd.set_option('display.max_colwidth', None)

# yoochoose clicks raw data
raw_data_path = Path('./datasets/yoochoose/yoochoose-clicks.dat')

df = pd.read_csv(raw_data_path, names=['session_id', 'timestamp', 'item_id', 'category'])

# reencode items
df['item_id'] = df['item_id'].astype('category')
df['item_id'] = df['item_id'].cat.codes
print('Max new cat id: ', df['item_id'].astype('int64').max())

grouped_df = df.groupby('session_id', as_index=False).agg({'item_id': list})

# get some sessions
grouped_df_slice = grouped_df.iloc[:1000]

# find sessions with exact length
sessions = grouped_df_slice[grouped_df_slice['item_id'].map(len) > 1]

# get targets as last elements in each session
# get session itself as new column
sessions['session'] = sessions['item_id'].apply(lambda x: x[:-1])
sessions['target'] = sessions['item_id'].apply(lambda x: x[-1])

# increase session length by adding max(len(sessions)) - len(target_session) elements to the end of the session
max_session_len = sessions['session'].apply(len).max()
sessions['session_padded'] = sessions['session'].apply(lambda x: x + [0] * (max_session_len - len(x)))
sessions['session_string'] = sessions['session_padded'].apply(lambda x: '[' + ', '.join([str(_) for _ in x]) + ']')
sessions['session_ready'] = sessions['session_string'] + sessions['target'].apply(lambda x: '|[' + str(x) + ']')

with open('./datasets/yoochoose/micro.train', 'w') as file:
    content = sessions['session_ready'].to_string(index=False)
    output = ''
    for line in content.split('\n'):
        output += line.strip() + '\n'
    file.write(output)
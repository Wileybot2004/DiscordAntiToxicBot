#import packages
import discord
from keras.layers import TextVectorization
import numpy as np
from keras import models
import pickle
#init

with open("X_values.pk1", "rb") as tmp:
    x = pickle.load(tmp)

model = models.load_model('toxic.h5')
MAX_FEATURES = 200000

vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')

vectorizer.adapt(x.values)

file = open("token.txt")
token = file.read()
file.close()

client = discord.Client(intents=discord.Intents.all())

output = 0
#main code

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    #inset toxic code here
    query = message.content
    text = vectorizer(query)
    output = (model.predict(np.expand_dims(text,0)) > 0.5).astype(int)
    if output.any() == 1:
        await message.reply("This message was deleted due to Toxicity")
        await message.delete()
    

client.run(token)
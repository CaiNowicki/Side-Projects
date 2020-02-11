import os
from twitchio.ext import commands
from dotenv import load_dotenv

load_dotenv()

bot = commands.Bot(
    # set up the bot
    irc_token=os.getenv('TMI_TOKEN'),
    client_id=os.getenv('CLIENT_ID'),
    nick=os.getenv('BOT_NICK'),
    prefix=os.getenv('BOT_PREFIX'),
    initial_channels=[os.getenv('CHANNEL')]
)


@bot.event
async def event_ready():
    print(f"{os.getenv('BOT_NICK')} is online!")
    ws = bot._ws
    await ws.send_privmsg(os.getenv('CHANNEL'), f"/me has landed")


@bot.event
async def event_message(ctx):
    if ctx.author.name.lower() == os.getenv('BOT_NICK').lower():
        return
    #  ctx.channel.send(ctx.content)
    await bot.handle_commands(ctx)

    if 'hello' in ctx.content.lower():
        await ctx.channel.send(f"Hi, @{ctx.author.name}!")

    if '!test' in ctx.content.lower():
        await ctx.channel.send('test passed weirdly')


@bot.event
async def on_message(ctx):
    if '!hello' in ctx.content.lower():
        await ctx.channel.send('hello again')
@bot.event
async def new(ctx):
    await ctx.channel.send('new test passed!')

if __name__ == "__main__":
    bot.run()

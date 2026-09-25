import json
import os

import discord

from discord.ext import commands

from config import DISCORD_TOKEN
from ai import ask_ai


# =========================
# Discord setup
# =========================

intents = discord.Intents.default()

intents.message_content = True

bot = commands.Bot(
    command_prefix="$",
    intents=intents
)


# =========================
# RAG settings
# =========================

RAG_SETTINGS_FILE = "rag_settings.json"


def load_rag_settings():

    if not os.path.exists(
        RAG_SETTINGS_FILE
    ):

        return {}


    try:

        with open(
            RAG_SETTINGS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"Could not load RAG settings: {e}"
        )

        return {}


def save_rag_settings():

    try:

        with open(
            RAG_SETTINGS_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                rag_settings,
                f,
                indent=4
            )

    except Exception as e:

        print(
            f"Could not save RAG settings: {e}"
        )


rag_settings = load_rag_settings()


def get_rag_enabled(guild_id):

    # RAG is enabled by default.
    return rag_settings.get(
        str(guild_id),
        True
    )


# =========================
# Discord events
# =========================

@bot.event
async def on_ready():

    activity = discord.Game(
        name="$gpt for Minecraft help"
    )

    await bot.change_presence(
        status=discord.Status.online,
        activity=activity
    )

    print(
        f"We have logged in as {bot.user}"
    )


@bot.event
async def on_message(message):

    if message.author == bot.user:
        return


    if message.content.startswith("$hello"):

        await message.channel.send(
            "Hello!"
        )


    # Required because we override on_message.
    await bot.process_commands(message)


# =========================
# Test command
# =========================

@bot.command()
async def test(ctx):

    await ctx.send(
        "Test command received!"
    )


# =========================
# RAG toggle command
# =========================

@bot.command()
async def rag(ctx, setting=None):

    # -------------------------
    # Require a server
    # -------------------------

    if ctx.guild is None:

        await ctx.send(
            "The RAG setting can only be changed "
            "inside a Discord server."
        )

        return


    guild_id = str(
        ctx.guild.id
    )


    # -------------------------
    # No argument
    # -------------------------

    if setting is None:

        enabled = get_rag_enabled(
            ctx.guild.id
        )

        status = (
            "enabled"
            if enabled
            else "disabled"
        )

        await ctx.send(
            f"Wiki retrieval is currently **{status}**."
        )

        return


    # -------------------------
    # Normalize setting
    # -------------------------

    setting = setting.lower()


    # -------------------------
    # Turn RAG on
    # -------------------------

    if setting == "on":

        rag_settings[guild_id] = True

        save_rag_settings()

        await ctx.send(
            "Wiki retrieval is now **enabled**."
        )

        return


    # -------------------------
    # Turn RAG off
    # -------------------------

    if setting == "off":

        rag_settings[guild_id] = False

        save_rag_settings()

        await ctx.send(
            "Wiki retrieval is now **disabled**."
        )

        return


    # -------------------------
    # Invalid argument
    # -------------------------

    await ctx.send(
        "Use `$rag on`, `$rag off`, "
        "or `$rag` to check the current setting."
    )


# =========================
# Long message helper
# =========================

async def send_long_message(ctx, text):

    """
    Discord messages are limited to 2,000 characters.
    """

    while len(text) > 2000:

        split_at = text.rfind(
            "\n",
            0,
            2000
        )


        if split_at <= 0:

            split_at = 2000


        await ctx.send(
            text[:split_at]
        )


        text = (
            text[split_at:]
            .lstrip()
        )


    if text:

        await ctx.send(
            text
        )


# =========================
# GPT command
# =========================

@bot.command()
async def gpt(ctx, *, prompt):

    async with ctx.typing():

        try:

            # -------------------------
            # Get server's RAG setting
            # -------------------------

            if ctx.guild is not None:

                use_rag = get_rag_enabled(
                    ctx.guild.id
                )

            else:

                # Default to RAG enabled in DMs.
                use_rag = True


            print(
                f"GPT request from "
                f"{ctx.author}: "
                f"RAG={'ON' if use_rag else 'OFF'}"
            )


            # -------------------------
            # Ask AI
            # -------------------------

            answer, sources = ask_ai(
                prompt,
                use_rag=use_rag
            )


            # -------------------------
            # Add Wiki sources
            # -------------------------

            if sources:

                source_text = (
                    "\n\n"
                    "**Minecraft Wiki retrieval:**\n"
                )


                for source in sources:

                    source_text += (

                        f"• **{source['title']}** "
                        f"(similarity: "
                        f"{source['score']:.4f})\n"

                        f"  <{source['url']}>\n"
                    )


                answer += source_text


            # -------------------------
            # Send answer
            # -------------------------

            await send_long_message(
                ctx,
                answer
            )


        except Exception as e:

            print(
                f"GPT command error: {e}"
            )


            await ctx.send(
                f"Error: {e}"
            )


# =========================
# Start bot
# =========================

bot.run(
    DISCORD_TOKEN
)
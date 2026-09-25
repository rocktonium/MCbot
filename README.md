I originally created this for personal use on my own Minecraft server but I wanted to publish for presentation.

To setup:
1.In .env file, input your Discord bot auth key and your OpenAI-compatible endpoint key. 
2.Currently the endpoint is set to OpenRouter with gpt-oss but can be changed in ai.py file to any endpoint and model you like.
3.Run bot.py! Currently bot is setup with a prebuilt wiki index but it may only have knowledge up to 1.20 based on the state of the wiki at the time. To rebuild and fetch the latest articles/Minecraft updates run the build_index.py file.

Commands:
$test Test if bot online/responsive
$hello Test command
$gpt Query the assistant
$rag Check if RAG is enabled
$rag on/off Enable/disable RAG

When RAG is disabled the bot still has knowledge of every minecraft block/item/entity up to 1.20. This is included in reference_context.txt and can be edited
When RAG is enabled the reference context is disabled so bot can search vector database.
gpt-oss-20b & 120b are sometimes prone to hallucinate so use a stronger model for best results, but I find it has the best performance for the price. Also recommend Deepseek 3.2 for a good cost

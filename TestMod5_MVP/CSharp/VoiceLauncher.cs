using System;
using System.IO;
using Barotrauma;
private static string ModLogPath = Path.Combine("LocalMods", "TestMod3", "mod.log"); // пример пути
// Появится в оверлее и в console.log
DebugConsole.NewMessage("[TestMod3] C# мод загружен!", Microsoft.Xna.Framework.Color.LawnGreen);

// Пишем в console.log через Logger (гарантированно в файл)
try
{
    GameMain.LuaCs.Logger.Log("[TestMod3] Logger: мод инициализирован.");
}
catch (Exception)
{
    // защитная обёртка — если Logger недоступен (маловероятно)
}

// Пишем в свой файл лога (путь можешь поменять)
try
{
    Directory.CreateDirectory(Path.GetDirectoryName(ModLogPath) ?? ".");
    File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [TestMod3] Мод загружен\n");
}
catch (Exception ex)
{
    DebugConsole.NewMessage("[TestMod3] Ошибка записи лога: " + ex.Message, Microsoft.Xna.Framework.Color.Orange);
}

namespace TestMod3
{
    // Появится в оверлее и в console.log
    DebugConsole.NewMessage("[TestMod3] C# мод загружен!", Microsoft.Xna.Framework.Color.LawnGreen);

    // Пишем в console.log через Logger (гарантированно в файл)
    try
    {
        GameMain.LuaCs.Logger.Log("[TestMod3] Logger: мод инициализирован.");
    }
    catch (Exception)
    {
        // защитная обёртка — если Logger недоступен (маловероятно)
    }

    // Пишем в свой файл лога (путь можешь поменять)
    try
    {
        Directory.CreateDirectory(Path.GetDirectoryName(ModLogPath) ?? ".");
        File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [TestMod3] Мод загружен\n");
    }
    catch (Exception ex)
    {
        DebugConsole.NewMessage("[TestMod3] Ошибка записи лога: " + ex.Message, Microsoft.Xna.Framework.Color.Orange);
    }
    class VoiceLauncher : IAssemblyPlugin
    {
        private static string ModLogPath = Path.Combine("LocalMods", "TestMod3", "mod.log"); // пример пути

        public VoiceLauncher()
        {
            // Появится в оверлее и в console.log
            DebugConsole.NewMessage("[TestMod3] C# мод загружен!", Microsoft.Xna.Framework.Color.LawnGreen);

            // Пишем в console.log через Logger (гарантированно в файл)
            try
            {
                GameMain.LuaCs.Logger.Log("[TestMod3] Logger: мод инициализирован.");
            }
            catch (Exception)
            {
                // защитная обёртка — если Logger недоступен (маловероятно)
            }

            // Пишем в свой файл лога (путь можешь поменять)
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(ModLogPath) ?? ".");
                File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [TestMod3] Мод загружен\n");
            }
            catch (Exception ex)
            {
                DebugConsole.NewMessage("[TestMod3] Ошибка записи лога: " + ex.Message, Microsoft.Xna.Framework.Color.Orange);
            }
        }

        public void Initialize()
        {
            DebugConsole.NewMessage("[TestMod3] Initialize() выполнен!", Microsoft.Xna.Framework.Color.LightBlue);
            GameMain.LuaCs.Logger.Log("[TestMod3] Initialize() выполнен!");
            File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [TestMod3] Initialize\n");
        }

        public void Dispose()
        {
            DebugConsole.NewMessage("[TestMod3] Dispose() выполнен!", Microsoft.Xna.Framework.Color.Red);
            GameMain.LuaCs.Logger.Log("[TestMod3] Dispose() выполнен!");
            File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [TestMod3] Dispose\n");
        }

        // Удобный метод для ошибок
        private void LogError(string msg)
        {
            DebugConsole.ThrowError("[TestMod3] ERROR: " + msg);
            GameMain.LuaCs.Logger.Log("[TestMod3] ERROR: " + msg);
            File.AppendAllText(ModLogPath, DateTime.Now.ToString("s") + " [ERROR] " + msg + "\n");
        }
    }
}

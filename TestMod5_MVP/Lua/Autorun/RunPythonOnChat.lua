-- путь к твоему EXE на клиенте
local exepath = "E:/SteamLibrary/steamapps/common/Barotrauma/LocalMods/TestMod3/Lua/Autorun/model/dist/VoiceModule/VoiceModule.exe"
local outputpath = "E:/SteamLibrary/steamapps/common/Barotrauma/LocalMods/TestMod3/Lua/Autorun/model/dist/VoiceModule/output.txt"
-- Таблица соответствий: текстовая команда -> OrderPrefab
local COMMAND_MAP = {
    ["Включение питания реактора"] = "operatereactor",      
    ["Выключение питания реактора"] = "operatereactor",      -- тот же prefab, но AI выключит сам
    ["Навигация к пункту назначения"] = "steer",            
    ["Навигация назад"] = "steer",
    ["Отстранение"] = "dismissed",                            
    ["Очистить элементы"] = "cleanupitems",
    ["Подождите"] = "wait",
    ["Ремонт механических систем"] = "repairmechanical",
    ["Ремонт повреждённых систем"] = "repairsystems",
    ["Ремонт электрических систем"] = "repairelectrical",
    ["Управление оружием"] = "operateweapons",
    ["Устранить утечки"] = "fixleaks",
    ["пожар"] = "extinguishfires",
    ["следование"] = "follow",
    ["сражение"] = "fightintruders",
    ["хил"] = "requestfirstaid"
}


-- Функция выдачи приказа ботам
local function giveBotsOrder(orderName)
    print('функция giveBotsOrder запустилась')
    -- тут корректно получаем первого клиента
    local client = Client.ClientList[1]
    print(client)
    player = client.Character
    print(player)

    -- Особый случай: лечение
    if orderName == "requestfirstaid" then
        local prefab = OrderPrefab.Prefabs[orderName]
        if not prefab then
            print(" OrderPrefab '" .. orderName .. "' не найден!")
            return
        end
        local order = Order(prefab, player, nil, 100.0, nil, nil)
        player.SetOrder(order, true)
        print( player.Name .. " запросил медицинскую помощь!")
        return
    end

    -- Общие приказы
    local orderPrefab = OrderPrefab.Prefabs[orderName]
    if not orderPrefab then
        print("OrderPrefab '" .. orderName .. "' не найден!")
        return
    end
    local botsFound = false
    for _, bot in pairs(Character.CharacterList) do
        if bot.IsBot and not bot.IsDead then
            botsFound = true
            local baseOrder = Order(orderPrefab, player, nil, player)
            local order = baseOrder:WithManualPriority(100)

            if bot.SetOrder then
                bot:SetOrder(order, true)
            elseif bot.AIController and bot.AIController.SetOrder then
                bot.AIController:SetOrder(order, true)
            else
                print(" Не удалось выдать приказ для " .. bot.Name)
            end

            print(bot.Name .. " получил приказ '" .. orderName .. "' от " .. player.Name)
        end
    end

    if not botsFound then
        print("Ни один бот не найден.")
    end
end

-- проверяем output.txt каждые полсекунды
Hook.Add("think", "CheckVoiceCommand", function()
    if File.Exists(outputpath) then
        local txt = File.Read(outputpath)
        File.Delete(outputpath)

        print("Голосовая команда: " .. txt)
        local ordertxt = COMMAND_MAP[txt]
        print(ordertxt)
        if ordertxt then
            giveBotsOrder(ordertxt)

        else
            print("Неизвестная команда: " .. txt)
        end
    end
end)

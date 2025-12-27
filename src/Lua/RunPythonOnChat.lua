local modname = "VoiceMod 3.0"
local moddir = nil

for pkg in ContentPackageManager.EnabledPackages.All do
    if pkg.Name == modname then
        moddir = pkg.Path:gsub("[/\\]filelist%.xml$", "")
        break
    end
end


local outputpath = moddir .. "/Lua/Autorun/model/main/output.txt"

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

    local SPEC_MAP = {
        ["капитан"] = "captain",     
        ["медик"] = "medicaldoctor",
        ["механик"] = "mechanic",
        ["инженер"] = "engineer",
        ["охранник"] = "securityofficer",
        ["помощник"] = "assistant",
        ["near"] = "near",
        ["all"] = "all" 
    }

-- Функция выдачи приказа ботам
local function giveBotsOrder(targetJob, orderName)
    print('функция giveBotsOrder запустилась')
    -- тут корректно получаем первого клиента
    local client = Client.ClientList[1]
    player = client.Character

    -- Особый случай: лечение
    if orderName == "requestfirstaid" then
        local prefab = OrderPrefab.Prefabs[orderName]
        if not prefab then
            print("OrderPrefab '" .. orderName .. "' не найден!")
            return
        end
        local order = Order(prefab, player, nil, nil):WithManualPriority(100)
        player:SetOrder(order, true)
        print(player.Name .. " запросил медицинскую помощь!")
        return
    end

    if targetJob == nil then
        targetJob = 'near'
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
            if targetJob == 'all' then
                botsFound = true
                -- Сброс ВСЕХ старых приказов
                local dismissed = OrderPrefab.Prefabs["dismissed"]
                local order = Order(dismissed, player, nil, player)
                bot:SetOrder(order:WithManualPriority(0), true)

                local baseOrder = Order(orderPrefab, player, nil, player)
                local order = baseOrder:WithManualPriority(100)
                if orderPrefab == "dismissed" then
                    order = baseOrder:WithManualPriority(0)
                end
                if bot.SetOrder then
                    bot:SetOrder(order, true)
                elseif bot.AIController and bot.AIController.SetOrder then
                    bot.AIController:SetOrder(order, true)
                else
                    print(" Не удалось выдать приказ для " .. bot.Name)
                end

                print(bot.Name .. " получил приказ '" .. orderName .. "' от " .. player.Name)
            end
            if (bot.Info.Job.Prefab.Identifier.Value == targetJob) then
                botsFound = true
                -- Сброс ВСЕХ старых приказов
                local dismissed = OrderPrefab.Prefabs["dismissed"]
                local order = Order(dismissed, player, nil, player)
                bot:SetOrder(order:WithManualPriority(0), true)

                local baseOrder = Order(orderPrefab, player, nil, player)
                local order = baseOrder:WithManualPriority(100)
                if orderPrefab == "dismissed" then
                    order = baseOrder:WithManualPriority(0)
                end

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
    end

    if not botsFound then
        print("Ни один бот не найден.")
    end
end


local function processCommand(txt, spec)
    print("Голосовая команда: " .. txt)
    print('spec ---' .. spec)
    local targetJob = SPEC_MAP[spec]
    local ordertxt = COMMAND_MAP[txt]
    if ordertxt then
        giveBotsOrder(targetJob, ordertxt)
    else
        print("Неизвестная команда: " .. txt)
    end
end

-- проверяем output.txt каждые полсекунды
Hook.Add("think", "CheckVoiceCommand", function()
    if File.Exists(outputpath) then
        local lines = {}
        for line in File.Read(outputpath):gmatch("([^\n]*)\n?") do
            if line ~= "" then table.insert(lines, line) end
        end
        File.Delete(outputpath)

        -- функция для удаления пробелов в начале и конце строки
        local function trim(s)
            return s:gsub("^%s+", ""):gsub("%s+$", "")
        end

        local txt  = lines[1] and trim(lines[1]) or nil   -- первая строка
        local spec = lines[2] and trim(lines[2]) or nil   -- вторая строка

        processCommand(txt, spec)
    end
end)
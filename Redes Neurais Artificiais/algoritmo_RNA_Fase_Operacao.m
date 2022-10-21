clear,clc
%% Carregamento dos dados
dados = xlsread('Dados anuais - sem grandes cidades.xlsx');
%dados = xlsread('Dados anuais - completo.xlsx');
%dados = xlsread('Dados anuais - somente grandes cidades.xlsx');

populacao = dados(:,1); 
PIB = dados(:,2); %mil
clientes = dados(:,3);
consumo = dados(:,4); %kWh
dados = [populacao PIB clientes consumo];
fator_de_correlacao_pearson = corrcoef(dados);

%Normalização dos dados
dados_normalizados = normalize(dados, 'range', [0 1]);
percentual_dados_validacao = 7; % em "%"

%Inserção do limiar -1
dados_normalizados = [-ones(length(dados_normalizados),1) dados_normalizados];

questionario_respota = 0;% Se deseja usar os dados aleatoriamente deixe a resposta do questionario como 0, caso deseje usar os valores fixos deixe como 1.
%Separação dos dados de treinamento e validação

for k=1 % criação do laço apenas para minimizar esta aba
    if questionario_respota == 0
        quantidade_de_dados_validacao = round(percentual_dados_validacao*length(dados_normalizados)/100); %D
        selecao_de_dados = randi([1 length(dados_normalizados)],1,quantidade_de_dados_validacao);
        dados_de_validacao_normalizados = dados_normalizados(selecao_de_dados,:);
        dados_de_treinamento_normalizados = dados_normalizados;
        dados_de_treinamento_normalizados(selecao_de_dados,:) = [];

        %Determinação das entradas e saídas (treinamento)
        dados_de_entrada = dados_de_treinamento_normalizados(:,1:size(dados_de_treinamento_normalizados,2)-1); %Matriz de entradas (E)
        dados_de_saida   = dados_de_treinamento_normalizados(:,size(dados_de_treinamento_normalizados,2)); %Vetor de saídas desejadas para cada linha dos dados de entrada
        quantidade_de_entradas = size(dados_de_entrada,2);
    end
    if questionario_respota == 1
        selecao_de_dados = [5,131,105,11,54,81,14,151,169,52,46,337,150,17,58]; %dados fixos
        quantidade_de_dados_validacao = size(selecao_de_dados,2);
        dados_de_validacao_normalizados = dados_normalizados(selecao_de_dados,:);
        dados_de_treinamento_normalizados = dados_normalizados;
        dados_de_treinamento_normalizados(selecao_de_dados,:) = [];

        %Determinação das entradas e saídas (treinamento)
        dados_de_entrada = dados_de_treinamento_normalizados(:,1:size(dados_de_treinamento_normalizados,2)-1); %Matriz de entradas (E)
        dados_de_saida   = dados_de_treinamento_normalizados(:,size(dados_de_treinamento_normalizados,2)); %Vetor de saídas desejadas para cada linha dos dados de entrada
        quantidade_de_entradas = size(dados_de_entrada,2);
    end
end

%Matrizes de pesos selecionadas
% selecione qual matriz de pesos irá utilzar
% para o conjunto sem as grandes cidades deixe a resposta como 1
% para o conjunto das grandes cidades deixe a resposta como 2
% para o conjunto total deixe a resposta como 3

resposta =1;
for k=1
    if resposta == 1
    vetor_pesos_entrada = [-0.21311544	0.951141541	0.438282892	0.557658334	0.675285217	0.000615026	0.02781304
    1.463098568	-0.582782161	0.344958418	0.335154174	-0.034379542	0.702180898	0.381914867
    -0.709677538	1.024805082	3.143339199	1.050480063	0.949015819	0.74835214	0.067666697
    -1.725152077	2.202995091	2.514105979	0.599800645	1.815066628	-0.682060807	0.791139315];
    vetor_pesos_saida = [3.12457052	-2.718150674 2.215369468 3.271380572 0.335456836 1.533040258 -1.0384194	0.075385116]';
    end
    if resposta == 2
    vetor_pesos_entrada = [0.786264446	-0.059134167	-0.514019633	0.089825062	-0.341442557	-0.53796762	-0.062933533
    1.051194621	-0.48226199	-0.689632412	0.92477965	-0.473618223	-0.709015071	-0.605349725
    0.592737123	-0.604539382	-0.936498914	0.582794708	-0.885489788	-0.897912779	-0.652552763
    1.079505058	-0.565267419	-0.874670645	0.272086645	-0.860998303	-0.892562993	-0.118041788];
    vetor_pesos_saida = [0.320242277	2.813235189	-0.962387658	-2.074777566	1.397189959	-1.660849814	-2.057943029	-0.740583456]';
    end
    if resposta == 3
    vetor_pesos_entrada = [-0.17328532	0.928492844	0.196702071	-0.089911705	-0.44118562	-0.278831899	0.239068427
    0.714443164	0.242662814	-1.905578671	0.320644603	-0.715233312	-0.64625689	-0.601623828
    0.121403797	1.313304396	-2.250053915	-0.729287385	-1.1821132	-0.682645762	-0.882901056
    0.280130907	0.660243321	-2.495873013	-0.700142538	-0.728679534	-0.774206817	-0.57834593];
    vetor_pesos_saida = [0.196882025	0.832203374	3.193615972	-4.313501325	-0.647030841	-2.613050608	-1.517358975	-1.484975934]';
    end
end
 % dados de validação
    dados_de_entrada_validacao = dados_de_validacao_normalizados(:,1:quantidade_de_entradas);
    dados_de_saida_validacao = dados_de_validacao_normalizados(:,quantidade_de_entradas+1);
%Parâmetros da função sigmóide
    alfa = 1;
    beta = 1;
%   DxN       DxE                      ExN
    I1 = dados_de_entrada_validacao * vetor_pesos_entrada;
%   DxN                           DxN
    Y1 = alfa.*(1./(1 + exp(-beta.*I1)));
%   1xD
    p = linspace(-1,-1,size(dados_de_validacao_normalizados,1));
%   Dx(N+1) Dx1 DxN
    Y1 =    [p' Y1];
%   Dx1 Dx(N+1)      (N+1)x1 
    I2 = Y1 * vetor_pesos_saida;
%   Dx1                            Dx1
    Y2 = alfa.*(1./(1 + exp(-beta.*I2)));

    %Desnormalização dos dados
    valor_real_desnormalizado = dados_de_saida_validacao*(max(dados(:,4))-min(dados(:,4))) + min(dados(:,4));
    Y2_desnormalizado = Y2*(max(dados(:,4))-min(dados(:,4))) + min(dados(:,4));
    % parâmetros de analise
    ERM = 100*sum(abs(valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado)/quantidade_de_dados_validacao;
    fprintf('ERM: %1.2f', ERM)
    disp('%')

    EQM_validacao = sum((dados_de_saida_validacao-Y2).^2)/quantidade_de_dados_validacao;
    fprintf('EQM: %1.5f \n', EQM_validacao)

    vetor_ERM = 100*(abs(valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado);
    desvio_padrao = sqrt(sum((vetor_ERM-ERM).^2)/quantidade_de_dados_validacao);
    fprintf('Desvio padrão: %1.2f', desvio_padrao)
    disp('%')
    
    fator_de_correlacao_pearson_validacao = corrcoef([valor_real_desnormalizado Y2_desnormalizado]);
    fprintf('Fator de correlação: %1.2f \n', fator_de_correlacao_pearson_validacao(1,2))

%% Plotar Graficos
        figure(1), plot(valor_real_desnormalizado/1E6,'r-o'), hold on, grid, plot(Y2_desnormalizado/1E6,'b-*'),legend('Saída Desejada','Saída Prevista'),xlabel('Amostras'),ylabel('Consumo anual (GWh)')


        
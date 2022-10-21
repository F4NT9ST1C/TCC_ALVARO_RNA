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

%Parâmetros de avaliação e controle
contador = 0;
tempo_segundos = 0;

%% Treinamento
numero_neuronios = 7; % inserir o número de neuronios da topologia vencedora 
numero_maximo_testes = 20;
questionario_respota = 1; % Se deseja usar os dados aleatoriamente deixe a resposta do questionario como 0 caso deseje usar os valores fixos deixe como 1.
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
        selecao_de_dados = [324,208,217,109,6,67,284,23,103,273,292,81,317,150,294,154,194,4,241,138]; %dados fixos
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
for testes = 1:numero_maximo_testes
    tic
    % Inicializando a matriz peso w1(ExN) e vetor coluna w2((N+1)x1) aleatoriamente
    % O vetor de pesos inicial é elevado à quinta só para garantir que seus elementos sejam pequenos
    vetor_pesos_entrada          = rand(quantidade_de_entradas,numero_neuronios).^5; %w1
    vetor_pesos_entrada_inicial  = vetor_pesos_entrada;
    vetor_pesos_entrada_atual    = vetor_pesos_entrada;
    vetor_pesos_entrada_anterior = vetor_pesos_entrada;
    vetor_pesos_saida            = rand(numero_neuronios+1,1).^5; %w2
    vetor_pesos_saida_inicial    = vetor_pesos_saida;
    vetor_pesos_saida_atual      = vetor_pesos_saida;
    vetor_pesos_saida_anterior   = vetor_pesos_saida;
    %Taxa de aprendizagem (ta) e precisão
    ta        = 0.0125;
    precisao  = 10^-7;

    %Numero máximo de épocas e época inicial
    max_ep = 2000;
    ep     = 2;

    %Parâmetros da função sigmóide
    alfa = 1;
    beta = 1;

    %Termo momentum
    gama = 0.8;

    %Erro Quadrático Médio inicial
    EQM = 0;
    clear eqm
    eqm(:,1) = .2;
    eqm(:,2) = .1;
    % Esses dois eqm anteriores foram declarados para que se tivesse 
    % pelo menos duas amostras para que se pudesse fazer o condicional 
    % do WHILE.

    % Laço principal
    while (abs(eqm(:,(ep)) - eqm(:,(ep-1))) > precisao && ep < max_ep)
        for k = 1:size(dados_de_treinamento_normalizados,1)
    % Fase Foward        
    %       Nx1         NxE                 Ex1               Nx1                           Nx1   (N+1)x1     
            I1 = vetor_pesos_entrada'*dados_de_entrada(k,:)'; Y1 = alfa.*(1./(1 + exp(-beta.*I1))); Y1 = [-1; Y1];
    %       1x1         1x(N+1)      (N+1)x1       1x1                            1x1
            I2 = vetor_pesos_saida'  *  Y1;         Y2 = alfa.*(1./(1 + exp(-beta.*I2)));

    % Início da retropropagação do erro (Fase backward)      
    % Derivada da função sigmóide: g'(z)=g(z)*[1-g(z)]
    %       Nx1                                Nx1                            Nx1
            a = (alfa*beta*(1./(1 + exp(-beta.*I1)))).*(1-(1./(1 + exp(-beta.*I1))));   %Derivada da função sigmóide em I1
    %       Nx1                                Nx1                            Nx1
            b = (alfa*beta*(1./(1 + exp(-beta.*I2)))).*(1-(1./(1 + exp(-beta.*I2))));   %Derivada da função sigmóide em I2
    %       1x1            1x1               1x1 
            delta2   = b*((dados_de_saida(k)-Y2));
    %       (N+1)x1                    (N+1)x1           1x1  (N+1)x1
            vetor_pesos_saida = vetor_pesos_saida + ((ta*delta2)*Y1)  + gama*(vetor_pesos_saida_atual - vetor_pesos_saida_anterior);
            vetor_pesos_saida_anterior = vetor_pesos_saida_atual;
            vetor_pesos_saida_atual = vetor_pesos_saida;

    %       Nx1       Nx1      1x1         Nx1
            delta1   = a .* (delta2*vetor_pesos_saida(2:(numero_neuronios+1)));
    %       ExN                          ExN                        Ex1             1xN     
            vetor_pesos_entrada = vetor_pesos_entrada + ta*(dados_de_entrada(k,:)'*delta1') + gama*(vetor_pesos_entrada_atual - vetor_pesos_entrada_anterior);
            vetor_pesos_entrada_anterior = vetor_pesos_entrada_atual;
            vetor_pesos_entrada_atual = vetor_pesos_entrada;

            EQ = 0.5*((dados_de_saida(k)-Y2)^2);
            EQM = EQM + EQ/size(dados_de_treinamento_normalizados,1);
        end
            ep = ep + 1;
            eqm(:,ep) = EQM;
            EQM = 0;
            %Serve somente para visualizar dinâmica do vetor de pesos da camada de saída
            %Cada linha representa a evolução do ajuste dos pesos relativos às saídas
            %da camada oculta e limiar de ativação.
            evolucao_vetor_pesos_saida(:,ep) = vetor_pesos_saida;
    end
%% Validação
    disp('Treinamento finalizado!')
    fprintf(1,'Número de épocas: %1.0f \n',ep)
    dados_de_entrada_validacao = dados_de_validacao_normalizados(:,1:quantidade_de_entradas);
    dados_de_saida_validacao = dados_de_validacao_normalizados(:,quantidade_de_entradas+1);

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
    ERM = 100*sum((valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado)/quantidade_de_dados_validacao;
    fprintf('ERM: %1.2f', ERM)
    disp('%')

    EQM_validacao = sum((dados_de_saida_validacao-Y2).^2)/quantidade_de_dados_validacao;
    fprintf('EQM: %1.5f \n', EQM_validacao)

    vetor_ERM = 100*(valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado;
    
    desvio_padrao = sqrt(sum((vetor_ERM-ERM).^2)/quantidade_de_dados_validacao);
    fprintf('Desvio padrão: %1.2f', desvio_padrao)
    disp('%')
    
    fator_de_correlacao_pearson_validacao = corrcoef([valor_real_desnormalizado Y2_desnormalizado]);
    fprintf('Fator de correlação: %1.2f \n', fator_de_correlacao_pearson_validacao(1,2))
    
    evolucao_desvio_padrao(testes,1) = desvio_padrao;
    evolucao_ERM(testes,1) = abs(ERM);
    evolucao_fator_correlacao(testes,1) = fator_de_correlacao_pearson_validacao(1,2);
    vetor_epocas(testes,1) = ep;
    evolucao_EQM(testes,1) =  EQM_validacao;
%% seleção das matrizes

    if testes == 1 % salvando o primeiro teste p/ comparação
    vetor_pesos_entrada_competicao = vetor_pesos_entrada;
    vetor_pesos_saida_competicao = vetor_pesos_saida;
    vetor_pesos__inicial_entrada_competicao = vetor_pesos_entrada_inicial;
    vetor_pesos__inicial_saida_competicao = vetor_pesos_saida_inicial;
    teste_vencedor_atual = 1;
    %parâmetros para plots de graficos
    eqm_plot = eqm;
    quantidade_de_epocas = ep;
    Y2_desnormalizado_plot = Y2_desnormalizado;
    evolucao_vetor_pesos_saida_competicao = evolucao_vetor_pesos_saida;
    else % Verificar o condicional após o segundo teste
    if evolucao_EQM(teste_vencedor_atual) > evolucao_EQM(testes)
    vetor_pesos_entrada_competicao = vetor_pesos_entrada;
    vetor_pesos_saida_competicao = vetor_pesos_saida;
    vetor_pesos__inicial_entrada_competicao = vetor_pesos_entrada_inicial;
    vetor_pesos__inicial_saida_competicao = vetor_pesos_saida_inicial;
    teste_vencedor_atual = testes;
    %parâmetros para plots de graficos
    eqm_plot = eqm;
    quantidade_de_epocas = ep;
    Y2_desnormalizado_plot = Y2_desnormalizado;
    evolucao_vetor_pesos_saida_competicao = evolucao_vetor_pesos_saida;
    end
    end
        contador = contador + 1;
        fprintf(1,'Teste %1.0f \n', contador)
        tempo_segundos = tempo_segundos + toc;
        vetor_testes(testes,1) = contador;
end
% Médias dos parâmetros gerais 
media_total_EQM = sum(evolucao_EQM)/numero_maximo_testes;
media_total_ERM = sum(evolucao_ERM)/numero_maximo_testes;
media_total_desvio_padrao = sum(evolucao_desvio_padrao)/numero_maximo_testes;
media_total_fator_correlacao = sum(evolucao_fator_correlacao)/numero_maximo_testes;
media_total_vetor_epocas = sum(vetor_epocas)/numero_maximo_testes;
%% Plotar Graficos
        figure(1), plot(valor_real_desnormalizado/1E6,'r-o'), hold on, grid, plot(Y2_desnormalizado_plot/1E6,'b-*')
        legend('Saída desejada(GWh)','Saída real(GWh)')
        figure(2),plot(3:quantidade_de_epocas,eqm_plot(3:length(eqm_plot))),grid
        legend('Erro Quadrático médio')
        figure(3)
        subplot(2,1,1),plot(valor_real_desnormalizado/1E6,'r-o'), hold on, grid, plot(Y2_desnormalizado_plot/1E6,'b-*'),legend('Saída desejada','Saída real'),xlabel('Amostras'),ylabel('Consumo anual (GWh)')
        subplot(2,1,2),plot(3:quantidade_de_epocas,eqm_plot(3:length(eqm_plot))),grid,xlabel('Épocas'),ylabel('Erro Quadrático médio') %remoção das duas primeiras épocas
        legenda = ['Desejado(GWh)    Saída Real(GWh)      Erro(%)'];
        analise = [valor_real_desnormalizado/1E6 Y2_desnormalizado_plot/1E6 100*(valor_real_desnormalizado/1E6-Y2_desnormalizado_plot/1E6)./valor_real_desnormalizado/1E6];
        figure4 = figure('WindowState','maximized');
        grafico = plot3(evolucao_vetor_pesos_saida(:,3:ep),3:ep,eqm(:,3:ep),'-o'); grid
        datatip(grafico(1),'DataIndex',ep,'Location','northwest');
        %stem3(evolucao_vetor_pesos_saida(1,3:ep),3:ep,eqm(:,3:ep))
        %comet3(evolucao_vetor_pesos_saida(:,3:ep),3:ep,eqm(:,3:ep)),grid
        xlabel('Vetor peso de saída'),ylabel('Época'),zlabel('EQM')

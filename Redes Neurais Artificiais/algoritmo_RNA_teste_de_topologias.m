clear,clc
%% Carregamento dos dados
dados = xlsread('Dados anuais - sem grandes cidades.xlsx');
%dados = xlsread('Dados anuais - completo.xlsx');
%dados = xlsread('Dados anuais - somente grandes cidades.xlsx');
populacao = dados(:,1); 
PIB = dados(:,2); %mil
clientes = dados(:,3);
consumo = dados(:,4); %kWh
%fator_ind = dados(:,6); %Fator que indica presença de indústria (1) ou ausência (0)
dados = [populacao PIB clientes consumo];
%dados = [populacao PIB clientes fator_ind consumo];
fator_de_correlacao_pearson = corrcoef(dados);

%Normalização dos dados
dados_normalizados = normalize(dados, 'range', [0 1]);
percentual_dados_validacao = 25; % em "%"

%Inserção do limiar -1
dados_normalizados = [-ones(length(dados_normalizados),1) dados_normalizados];

%Parâmetros de avaliação e controle
contador = 0;
tempo_segundos = 0;

%% Treinamento
%Método de kolmogorov: N = 2*(número de entradas)+1
numero_minimo_neuronios = 3;
numero_maximo_neuronios = 7;
numero_maximo_testes = 1000;
passo = 1;

for testes = 1:numero_maximo_testes
    %Separação dos dados de treinamento e validação
    quantidade_de_dados_validacao = round(percentual_dados_validacao*length(dados_normalizados)/100); %D
    selecao_de_dados = randi([1 length(dados_normalizados)],1,quantidade_de_dados_validacao);
    dados_de_validacao_normalizados = dados_normalizados(selecao_de_dados,:);
    dados_de_treinamento_normalizados = dados_normalizados;
    dados_de_treinamento_normalizados(selecao_de_dados,:) = [];

    %Determinação das entradas e saídas (treinamento)
    dados_de_entrada = dados_de_treinamento_normalizados(:,1:size(dados_de_treinamento_normalizados,2)-1); %Matriz de entradas (E)
    dados_de_saida   = dados_de_treinamento_normalizados(:,size(dados_de_treinamento_normalizados,2)); %Vetor de saídas desejadas para cada linha dos dados de entrada
    quantidade_de_entradas = size(dados_de_entrada,2);
    for quantidade_de_neuronios_camada_oculta = numero_minimo_neuronios:passo:numero_maximo_neuronios
        tic
        %Pré-alocação das matrizes utilizadas
        if quantidade_de_neuronios_camada_oculta == numero_minimo_neuronios && testes == 1
            pre_alocacao = zeros(length(numero_minimo_neuronios:passo:numero_maximo_neuronios),numero_maximo_testes);
            evolucao_desvio_padrao = pre_alocacao;
            vetor_neuronios = pre_alocacao;
            evolucao_ERM = pre_alocacao;
            evolucao_EQM = pre_alocacao;
            evolucao_fator_correlacao = pre_alocacao;
        end
        
        % Inicializando a matriz peso w1(ExN) e vetor coluna w2((N+1)x1) aleatoriamente
        % O vetor de pesos inicial é elevado à quinta só para garantir que seus elementos sejam pequenos
        vetor_pesos_entrada          = rand(quantidade_de_entradas,quantidade_de_neuronios_camada_oculta).^5; %w1
        vetor_pesos_entrada_inicial  = vetor_pesos_entrada;
        vetor_pesos_entrada_atual    = vetor_pesos_entrada;
        vetor_pesos_entrada_anterior = vetor_pesos_entrada;
        vetor_pesos_saida            = rand(quantidade_de_neuronios_camada_oculta+1,1).^5; %w2
        vetor_pesos_saida_inicial    = vetor_pesos_saida;
        vetor_pesos_saida_atual      = vetor_pesos_saida;
        vetor_pesos_saida_anterior   = vetor_pesos_saida;
        
        %Taxa de aprendizagem (ta) e precisão
        ta        = 0.1;
        precisao  = 10^-6;

        %Numero máximo de épocas e época inicial
        max_ep = 2000;
        ep     = 2;

        %Parâmetros da função sigmóide
        alfa = 1;
        beta = 1;
        
        %Termo momentum
        gama = 0.80;

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
                delta1   = a .* (delta2*vetor_pesos_saida(2:(quantidade_de_neuronios_camada_oculta+1)));
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
        ERM = 100*sum(abs(valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado)/quantidade_de_dados_validacao;
        fprintf('ERM: %1.2f', ERM)
        disp('%')
        
        EQM_validacao = sum((dados_de_saida_validacao-Y2).^2)/quantidade_de_dados_validacao;
        fprintf('EQM: %1.5f \n', EQM_validacao)
        
        vetor_ERM = 100*abs(valor_real_desnormalizado-Y2_desnormalizado)./valor_real_desnormalizado;
        desvio_padrao = sqrt(sum((vetor_ERM-ERM).^2)/quantidade_de_dados_validacao);
        fprintf('Desvio padrão: %1.2f', desvio_padrao)
        disp('%')
        
        fator_de_correlacao_pearson_validacao = corrcoef([valor_real_desnormalizado Y2_desnormalizado]);
        fprintf('Fator de correlação: %1.2f \n', fator_de_correlacao_pearson_validacao(1,2))
        fprintf('Quantidade de neurônios: %1.0f \n', quantidade_de_neuronios_camada_oculta)
     
        vetor_neuronios(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = quantidade_de_neuronios_camada_oculta;
        evolucao_desvio_padrao(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = desvio_padrao;
        evolucao_ERM(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = ERM;
        evolucao_EQM(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = EQM_validacao;
        evolucao_fator_correlacao(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = fator_de_correlacao_pearson_validacao(1,2);
        vetor_epocas(quantidade_de_neuronios_camada_oculta-(numero_minimo_neuronios-1),testes) = ep;
        
        contador = contador + 1;
        fprintf(1,'Teste %1.0f \n', contador)
        tempo_segundos = tempo_segundos + toc;
    end
end

%Remoção das linhas de zero devido não utilizar procura sequencial da
%quantidade de neurônios na camada escondida
linhas_a_remover = find(sum(vetor_neuronios,2) == 0);
vetor_neuronios(linhas_a_remover,:) = [];
linhas_a_remover = find(sum(evolucao_desvio_padrao,2) == 0);
evolucao_desvio_padrao(linhas_a_remover,:) = [];
linhas_a_remover = find(sum(evolucao_ERM,2) == 0);
evolucao_ERM(linhas_a_remover,:) = [];
linhas_a_remover = find(sum(evolucao_EQM,2) == 0);
evolucao_EQM(linhas_a_remover,:) = [];
linhas_a_remover = find(sum(evolucao_fator_correlacao,2) == 0);
evolucao_fator_correlacao(linhas_a_remover,:) = [];
linhas_a_remover = find(sum(vetor_epocas,2) == 0);
vetor_epocas(linhas_a_remover,:) = [];

%% Definição da topologia vencedora
%1º critério: 
%Desempate no 1º critério: menor quantidade de neurônios
%2º critério: 
%Desempate no 2º critério: menor quantidade de neurônios

%Ordenação ascendente dos resultados para cada topologia
%evolucao_ERM = sort(abs(evolucao_ERM),2); %O "2" significa ordenação apenas das linhas
%evolucao_EQM = sort(abs(evolucao_EQM),2); %O "2" significa ordenação apenas das linhas
%evolucao_desvio_padrao = sort(abs(evolucao_desvio_padrao),2); %O "2" significa ordenação apenas das linhas
%evolucao_fator_correlacao = sort(abs(evolucao_fator_correlacao),2); %O "2" significa ordenação apenas das linhas

%Colocando os dados em modo absoluto (módulo)
evolucao_ERM = abs(evolucao_ERM);
evolucao_EQM = abs(evolucao_EQM);
evolucao_desvio_padrao = abs(evolucao_desvio_padrao);
evolucao_fator_correlacao = abs(evolucao_fator_correlacao);

%Critério do somatório total
soma_ERM_topologia = sum(evolucao_ERM,2); %soma o ERM para cada topologia (linha da matriz)
[menor_valor,~] = min(soma_ERM_topologia); %procura o menor valor de ERM
[linha,~] = find(soma_ERM_topologia == menor_valor); %Identifica a linha (topologia) com menor valor
if passo == 1
    vencedora = linha + (numero_minimo_neuronios-1);
else
    vencedora = passo*linha;
end
topologia_vencedora_ERM = vencedora; %Definição da topologia vencedora
media_ERM_topologia = mean(evolucao_ERM,2); %média do ERM para cada topologia (linha da matriz)

soma_EQM_topologia = sum(evolucao_EQM,2);
[menor_valor,~] = min(soma_EQM_topologia);
[linha,~] = find(soma_EQM_topologia == menor_valor);
if passo == 1
    vencedora = linha + (numero_minimo_neuronios-1);
else
    vencedora = passo*linha;
end
topologia_vencedora_EQM_somatorio = vencedora;
media_EQM_topologia = mean(evolucao_EQM,2);

soma_desvio_padrao_topologia = sum(evolucao_desvio_padrao,2);
[menor_valor,~] = min(soma_desvio_padrao_topologia);
[linha,~] = find(soma_desvio_padrao_topologia == menor_valor);
if passo == 1
    vencedora = linha + (numero_minimo_neuronios-1);
else
    vencedora = passo*linha;
end
topologia_vencedora_desvio_padrao = vencedora;
media_desvio_padrao_topologia = mean(evolucao_desvio_padrao,2);

soma_fator_correlacao_topologia = sum(evolucao_fator_correlacao,2);
[maior_valor,~] = max(soma_fator_correlacao_topologia);
[linha,~] = find(soma_fator_correlacao_topologia == maior_valor);
if passo == 1
    vencedora = linha + (numero_minimo_neuronios-1);
else
    vencedora = passo*linha;
end
topologia_vencedora_fator_correlacao = vencedora;
media_fator_correlacao_topologia = mean(evolucao_fator_correlacao,2);

%Critério do melhor valor para cada teste
matriz_decisao = zeros(size(evolucao_EQM,1),size(evolucao_EQM,2));
for i = 1:length(evolucao_EQM)
    [menor_valor,~] = min(evolucao_EQM(:,i));
    [linha,~] = find(evolucao_EQM(:,i) == menor_valor);
    matriz_decisao(linha,i) = 1;
end
somatorio_competicao = sum(matriz_decisao,2);
topologia_vencedora_EQM_competicao = min(find(somatorio_competicao == max(somatorio_competicao)) + (numero_minimo_neuronios-1));
%%
% % %% Gráficos
% figure(1)
% subplot(2,2,1),plot(vetor_neuronios,evolucao_ERM,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('ERM (%)')
% hold on, plot(numero_minimo_neuronios:numero_maximo_neuronios,media_ERM_topologia,'s','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','r')
% subplot(2,2,2),plot(vetor_neuronios,evolucao_EQM,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('EQM')
% hold on, plot(numero_minimo_neuronios:numero_maximo_neuronios,media_EQM_topologia,'s','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','r')
% subplot(2,2,3),plot(vetor_neuronios,evolucao_fator_correlacao,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('Fator de correlação')
% hold on, plot(numero_minimo_neuronios:numero_maximo_neuronios,media_fator_correlacao_topologia,'s','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','r')
% subplot(2,2,4),plot(vetor_neuronios,evolucao_desvio_padrao,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('Desvio padrão (%)')
% hold on, plot(numero_minimo_neuronios:numero_maximo_neuronios,media_desvio_padrao_topologia,'s','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','y','MarkerFaceColor','r')
%%
% figure(2)
% subplot(4,1,1),hist(evolucao_ERM'),grid,xlabel('Erro relativo médio (%)'),ylabel('Quantidade de amostras'),colorbar
% subplot(4,1,2),hist(evolucao_EQM'),grid,xlabel('EQM'),ylabel('Quantidade de amostras'),colorbar
% subplot(4,1,3),hist(evolucao_fator_correlacao'),grid,xlabel('Fator de correlação (%)'),ylabel('Quantidade de amostras'),colorbar
% subplot(4,1,4),hist(evolucao_desvio_padrao'),grid,xlabel('Desvio padrão (%)'),ylabel('Quantidade de amostras'),colorbar
% 
% % figure(3)
% % subplot(1,4,1),boxplot(evolucao_ERM',(numero_minimo_neuronios:numero_maximo_neuronios)'),grid,xlabel('Quantidade de neurônios'),ylabel('Erro relativo médio (%)')
% % subplot(1,4,2),boxplot(evolucao_EQM',(numero_minimo_neuronios:numero_maximo_neuronios)'),grid,xlabel('Quantidade de neurônios'),ylabel('EQM')
% % subplot(1,4,3),boxplot(evolucao_fator_correlacao',(numero_minimo_neuronios:numero_maximo_neuronios)'),grid,xlabel('Quantidade de neurônios'),ylabel('Fator de correlação (%)')
% % subplot(1,4,4),boxplot(evolucao_desvio_padrao',(numero_minimo_neuronios:numero_maximo_neuronios)'),grid,xlabel('Quantidade de neurônios'),ylabel('Desvio padrão (%)')
% 
% figure(4)
% subplot(2,1,1),plot(valor_real_desnormalizado/1E6,'r-o'), hold on, grid, plot(Y2_desnormalizado/1E6,'b-*'),legend('Saída desejada','Saída real'),xlabel('Amostras'),ylabel('Consumo anual (GWh)')
% subplot(2,1,2),plot(3:ep,eqm(3:length(eqm))),grid,xlabel('Épocas'),ylabel('Erro Quadrático médio') %remoção das duas primeiras épocas
% legenda = ['Desejado(GWh)    Saída Real(GWh)      Erro(%)'];
% analise = [valor_real_desnormalizado/1E6 Y2_desnormalizado/1E6 100*(valor_real_desnormalizado/1E6-Y2_desnormalizado/1E6)./valor_real_desnormalizado/1E6];
%%
quantidade_neuronios = size(soma_EQM_topologia,1);
figure(1)
subplot(1,2,1),plot(vetor_neuronios,evolucao_desvio_padrao,'ob','LineWidth',1),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('Desvio padrão (%)')
subplot(1,2,2),plot(vetor_neuronios(:,1),media_desvio_padrao_topologia,'*r','LineWidth',7),hold on,grid,plot(topologia_vencedora_desvio_padrao,min(media_desvio_padrao_topologia),'*k','LineWidth',8),grid,xlabel('Quantidade de neurônios'),ylabel('Média do Desvio padrão (%) para cada topologia');
figure(2)
subplot(1,2,1),plot(vetor_neuronios,evolucao_fator_correlacao,'ob','LineWidth',1),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('Coeficiente de correlação de Pearson');
subplot(1,2,2),plot(vetor_neuronios(:,1),media_fator_correlacao_topologia,'*r','LineWidth',7),hold on,grid,plot(topologia_vencedora_fator_correlacao,max(media_fator_correlacao_topologia),'*k','LineWidth',8),grid,xlabel('Quantidade de neurônios'),ylabel('Média do coeficiente de correlação de Pearson para cada topologia');
figure(3)
subplot(1,2,1),plot(vetor_neuronios,evolucao_ERM,'ob','LineWidth',1),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('ERM (%)')
subplot(1,2,2),plot(vetor_neuronios(:,1),media_ERM_topologia,'*r','LineWidth',7),hold on,grid,plot(topologia_vencedora_ERM,min(media_ERM_topologia),'*k','LineWidth',8),grid,xlabel('Quantidade de neurônios'),ylabel('Média do ERM (%) para cada topologia')
figure(4)
subplot(1,2,1),plot(vetor_neuronios,evolucao_EQM,'ob','LineWidth',1),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('Erro quadratico médio ')
subplot(1,2,2),plot(vetor_neuronios(:,1),media_EQM_topologia,'*r','LineWidth',7),hold on,grid,plot(topologia_vencedora_EQM_somatorio,min(media_EQM_topologia),'*k','LineWidth',8),grid,xlabel('Quantidade de neurônios'),ylabel('Média do EQM para cada topologia')

figure(5)
subplot(1,3,1),plot(vetor_neuronios,evolucao_desvio_padrao,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('Desvio padrão (%)')
subplot(1,3,2),plot(vetor_neuronios,evolucao_ERM,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('ERM (%)')
subplot(1,3,3),plot(vetor_neuronios,evolucao_fator_correlacao,'*'),grid,xlabel('Quantidade de neurônios'),ylabel('Fator de correlação')

figure(6)
subplot(3,1,1),hist(evolucao_desvio_padrao'),grid,xlabel('Desvio padrão (%)'),ylabel('Quantidade de amostras'),colorbar
subplot(3,1,2),hist(evolucao_ERM'),grid,xlabel('Erro relativo médio (%)'),ylabel('Quantidade de amostras'),colorbar
subplot(3,1,3),hist(evolucao_fator_correlacao'),grid,xlabel('Fator de correlação (%)'),ylabel('Quantidade de amostras'),colorbar

figure(7)
subplot(1,2,1),plot(vetor_neuronios,vetor_epocas,'ob','LineWidth',1),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('Quantidade de épocas por treinamento.')
subplot(1,2,2),plot(vetor_neuronios(:,1),mean(vetor_epocas,2),'*r','LineWidth',7),hold on,grid,xlabel('Quantidade de neurônios'),ylabel('Média da quantidade de épocas por treinamento.')
%%
clc
fprintf(1,'Tempo total de simulação (minutos): %1.2f \n', tempo_segundos/60)
fprintf(1,'Tempo relativo de simulação (minutos/topologia): %1.2f \n', (tempo_segundos/60)/(numero_maximo_neuronios-numero_minimo_neuronios+1))
fprintf(1,'Tempo relativo de simulação (segundos/teste): %1.3f \n', (tempo_segundos)/(numero_maximo_testes*(numero_maximo_neuronios-numero_minimo_neuronios+1)))

##############  
#  jMef 0.6  #  English version
##############  


# Importacion de librerias 

import numpy as np
from numpy import exp, sin, cos, log  # para ponerlo asi en las xgrande & temperatura
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import lstsq

from tkinter import *
from tkinter import ttk, messagebox, filedialog

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d # provisonalmente. No hare 3D en version final
matplotlib.use('TkAgg')  # mac requiere hacer esto explicitamente
from os import path, kill, getppid, _exit
from copy import deepcopy
import signal


class Nodo:
    def __init__(self, x=0., y=0.):
        # solo almacena info, no se necesita mas
        self.x, self.y = x,y
        self.ux, self.uy = 0., 0.
        self.elemn=[]
        self.pseudo_n_ext = [0]
        self.apoyo_incl= 0. # si tiene apoyo inclinado, el angulo
        self.sigma = np.array([0., 0., 0., 0])  # xx, yy, xy, vm
        # tensiones principales:
        self.s_prles=[0, [0,0], 0, [0,0]]
        


class Elem:
    '''
    Geometry function coefficients (H), 
    normalized node coord (-1,1 or similar),
    normalized coord of Gauss points and its weights,
    ->  are all stored as class attributes.
    They're not to be instantiated as they are the same for each triangle 6.
    '''
    
    # Coef de las ff geometricas H. Hay que multiplicar asi:
    # [1, x_, y_, x_**2, x_*y_, etc] *[coef[][]] = [H0(x_y_), H1(x_y_), etc ] 
    # para obtener los valores de las ff H en (x_,y_) ¡que son coor normalizadas!
    
    r3=np.sqrt(3.)
    coefH=np.array([
    [ 1.00,    0.00,    0.00,    0.00,    0.00,   0.00 ],
    [ 0.00,    2.00,   -0.50,    0.00,    0.50,  -2.00 ],
    [  -r3,    2/r3,  -0.5/r3,   0.00, -0.5/r3,   2/r3 ],
    [ 0.00,    0.00,    0.50,   -1.00,    0.50,   0.00 ],
    [ 0.00,   -2/r3,    1/r3,    0.00,   -1/r3,   2/r3 ],
    [ 2/3.,   -2/3.,    1/6.,    1/3.,    1/6.,   -2/3.]   ])


    # Coordenadas normalizadas de los nodos 
    # (lo que tenga _ sera indicativo de normalizado, del espacio psi,eta)
    a=r3/2.
    x_ = np.array([0.0, 0.5, 1.0, 0.0, -1.0, -0.5])
    y_ = np.array([0.0, a,   2*a, 2*a,  2*a,  a])
    
    # Puntos de Gauss y sus pesos
    x_Gaus= [0.000000000000000,  0.337845472747894,  0.725271359470688, 
             0.000000000000000, -0.725271359470688, -0.337845472747894] 
    y_Gaus= [0.317229309127397,  0.959645363743758,  1.573436153005179,  
             1.544810887650238,  1.573436153005179,  0.959645363743758]
    w_Gaus= [0.190442006391806,  0.386908262797819,  0.190442006391806,
             0.386908262797819,  0.190442006391806,  0.386908262797819]

    # Las triangulaciones para dibujar tensiones etc seran atributos de clase,
    # pero quiero subdiv menor de 5 si ya hay muchos elems lo que no se hasta
    # que vaya a ejecutar => lo asigno con una funcion externa cuando vaya a
    # dibujar (asigna_tri_pintar). Aqui lo defino para que no quede raro que 
    # aparezca el atributo tri_pintar fuera, aunque creo que no daría error.
    tri_pintar=[]

    # Los vecinos se asignan en el programa, en un analisis sabidos todos los datos:
    # vecinos = [ [elem_vecino, lado_suyo], ...*3 ] para mis lados 0,1,2.
    # PERO si es un borde libre sera elem_vecino=-1 y lado_suyo= -1.
    # Podria usar la posicion lado_suyo para indicar si hay cczona, pero
    # creo que es innecesario.
    # Las cc en puntuales no se tienen en cuenta aqui (lado_suyo=-1).
    
    # En cczona[] guardo las cc de zona con dos terminos mas q el gui:
    # un string 'zu_imp', 'zkas', o 'xraya' que indica que tipo de cc es,
    # y un entero, el lado del elemento
    
    def __init__(self, mat=0, nodse=[]) :
        self.mat= mat
        self.nodse = nodse    # lista[] con los nodos del elemento
        self.xi = []
        self.yi = []    # coordenadas xy de los nodos del elemento
        self.cczona=[]
        self.vecinos= [ [-1,-1], [-1,-1], [-1,-1] ]  # ver arriba
        self.huevo=[[], [], []]
        # huevo[0] tiene los nodos que derivaran del elem
        # huevo[1], huevo[2] tienen las x,y de esos nodos
        self.sigma = np.zeros((6,4))
        # para cada pto de Gauss: s_xx, s_yy, s_xy, s_vm

    def dimeHi(self, psi,eta):
        # devuelve una lista con el valor de las Hi en el pto (psi,eta)
        pol=np.array([1., psi, eta, psi**2, psi*eta, eta**2])
        Hi= np.matmul(pol, self.coefH)
        return(Hi)

    def dimexy(self, psi,eta):
        # devuelve x,y correspondiente a psi,eta
        Hi=self.dimeHi(psi,eta)
        x = np.matmul( self.xi, Hi )
        y = np.matmul( self.yi, Hi )
        return(x,y)
    
    def dimeJacoss(self, psi,eta):
        # devuelve el jaco en ese pto y las derivadas de las 6 Hi
        pol_psi=np.array([0., 1., 0., 2*psi, eta,  0.])
        pol_eta=np.array([0., 0., 1.,   0.,  psi, 2*eta])
        Hi_psi= np.matmul(pol_psi, self.coefH)
        Hi_eta= np.matmul(pol_eta, self.coefH)
        x_psi= np.matmul(self.xi, Hi_psi)
        x_eta= np.matmul(self.xi, Hi_eta)
        y_psi= np.matmul(self.yi, Hi_psi)
        y_eta= np.matmul(self.yi, Hi_eta)
        jaco=np.array([[x_psi, x_eta],[y_psi, y_eta]]) 
        return(jaco, Hi_psi, Hi_eta)
    
    def pinta_elem(self, colorin='', tipolin='', ancholin=0.8, npl=17, alfalin=1):
        # dibuja las lineas (curvas o no) de los lados sin deformar, nada mas
        # para pintar deformadas uso zona()
        # npl= numero de puntos trazados de cada lado. Con impar incluye psi =0.5
        
        def puntea_lado(self, xi_, yi_, x, y):
            # no se usa si npl=3
            for i in range(npl):
                a,b= self.dimexy(xi_[i], yi_[i])
                x.append(a)
                y.append(b)
            return()

        tipolin= '-' if tipolin == ''  else tipolin

        if npl > 3 :
            x, y = [], []
            r3=np.sqrt(3.)
            xi_ , yi_ = np.linspace(-1,1,npl), [r3]*npl
            puntea_lado(self, xi_, yi_, x, y)

            a= np.array([-1, -r3])
            b=np.linspace(0,1,npl)
            xi_, yi_ = [], []
            for i in range(npl):
                v = np.array([1,r3]) + a*b[i]
                xi_.append(v[0])
                yi_.append(v[1])
            puntea_lado(self, xi_, yi_, x, y)

            a= np.array([-1, r3])
            xi_, yi_ = [], []
            for i in range(npl):
                v = a*b[i]
                xi_.append(v[0])
                yi_.append(v[1])
            puntea_lado(self, xi_, yi_, x, y)
            plt.plot(x,y, tipolin, linewidth=ancholin, color=colorin, alpha=alfalin)
        
        elif npl==3: # lo anterior funciona tb pero es lento. Mejor:
            plt.plot(self.xi + [self.xi[0]], self.yi + [self.yi[0]], 
                        tipolin, linewidth=ancholin, color=colorin, alpha=alfalin)
        
        
        return()



class Dominio:
    
    def __init__(self, nodos={}, elems={}, ctes={}, tplana=1):
        self.nodos=nodos
        self.elems=elems
        self.ctes=ctes # diccionario de materiales: {imat:[E,nu,alfaT], imat:[E,nu.alfaT]...}
        self.tplana=tplana
        self.extremosx, self.extremosy = [0.,0.],[0.,0.]
        self.anchox, self.anchoy = 0., 0.
        self.contornos=[]

    def pon_basicos(self):
        # asignar nodo.elemn[] & elem.xi[], elem.yi[]
        for nodo in self.nodos.values():
            nodo.elemn=[]
        for el in self.elems.values():
            el.xi, el.yi = [], []
        for iel,el in self.elems.items():
            for inodo in el.nodse:
                self.nodos[inodo].elemn.append(iel)
                el.xi.append(self.nodos[inodo].x)
                el.yi.append(self.nodos[inodo].y)
    
    def pon_extremos(self):
        # asignar extremos etc, usadas en dibujar y quiza en algo mas
        extremosx, extremosy = [0.,0.], [0.,0.] # cada uno [min, max]
        for n in self.nodos.values():
            if n.x < self.extremosx[0]: self.extremosx[0]=n.x
            if n.x > self.extremosx[1]: self.extremosx[1]=n.x
            if n.y < self.extremosy[0]: self.extremosy[0]=n.y
            if n.y > self.extremosy[1]: self.extremosy[1]=n.y
        self.anchox = self.extremosx[1]-self.extremosx[0]
        self.anchoy = self.extremosy[1]-self.extremosy[0]
    
    
    def pon_vecinos(self):
        # Asigna elem.vecinos(). Es del tipo:
        # elem.vecinos = [ [elem_vecino, lado_suyo], ...*3 ] para mis lados 0,1,2.
        # Si es un borde libre sera elem_vecino=-1 & lado_suyo= -1 (podria usar
        # la posicion de lado_suyo para indicar si hay cczona, pero creo que 
        # sera innecesario.
        # Las cc en desplazamientos no se tienen en cuenta aqui (lado_suyo=[]).
        # Inicializados elem.vecinos[] en elem.init =[[-1,-1],[-1,-1],[-1,-1]]
        
        for iel, el in self.elems.items():
            el.vecinos= [  [-1,[]] , [-1,[]] , [-1,[]]  ] # por si acaso
            lados=[ [el.nodse[0], el.nodse[2]], 
                    [el.nodse[2], el.nodse[4]],
                    [el.nodse[4], el.nodse[0]] ]
            for ilado, lado in enumerate(lados):
                for nodo_esq in lado:
                    e_candidatos = self.nodos[nodo_esq].elemn
                    for e_candi in e_candidatos:
                        lados_e_candi=[ [self.elems[e_candi].nodse[2], self.elems[e_candi].nodse[0]], 
                                        [self.elems[e_candi].nodse[4], self.elems[e_candi].nodse[2]],
                                        [self.elems[e_candi].nodse[0], self.elems[e_candi].nodse[4]] 
                                       ]
                        for ilado_e_candi, lado_e_candi in enumerate(lados_e_candi):
                            if lado == lado_e_candi:
                                el.vecinos[ilado]= [e_candi, ilado_e_candi]
                            
            #print('\nvecinos del elemento ', iel,' : ', el.vecinos)
    
    def pon_contornos(self):
        # Requiere pon_vecinos() primero.
        # Preveo usarlo para las pseudo_n_ext, preferentemente las del
        # DEL CASO BASE para dibujar, pero puede usarse el de un refinado.
        # Un contorno es parecido a un objeto zona[]: consta de tres nodos y 
        # adicionalmente info de a que lado de que elemento corresponde.
        # contorno = [[n1,n2,n3] , [elem, lado_elem]]
        self.contornos=[]
        for iel, el in self.elems.items():
            n_vecinos=0
            for ilado, lado in  enumerate(el.vecinos):
                if lado[0] == -1:  # es contorno
                    if ilado==0:
                        ni=el.nodse[0:3]
                        kk=[iel, ilado]
                    elif ilado==1:
                        ni=el.nodse[2:5]
                        kk=[iel, ilado]
                    else:
                        ni=[el.nodse[4], el.nodse[5],el.nodse[0]]
                        kk=[iel, ilado]
                    self.contornos.append([ni,kk])
    
    def pon_pseudo_n_ext(self):
        # Requiere pon_contornos() primero
        # Asigna una pseudo_n_ext a los nodos que esten en contorno
        for c in self.contornos:
            xi, yi = [], []
            for inodo in c[0]:
                xi.append(self.nodos[inodo].x)
                yi.append(self.nodos[inodo].y)
            zona=Zona(xi,yi)
            psi=0.
            for inodo in c[0]:
                n_ext1 = zona.n_ext(psi)[0]
                b= self.nodos[inodo].pseudo_n_ext
                if len(b):  # si ya tiene algo es una esquina. Promediar.
                    a  = b + n_ext1
                    a /= np.linalg.norm(a)
                else:
                    a = n_ext1
                self.nodos[inodo].pseudo_n_ext=a
                psi += 0.5
        

class CCpunto:
    # son cc que no cambian al refinar
    def __init__(self, u_imp={}, kas={}, fuerzas={}, xgrande=[], temperatura=[]):
        self.u_imp = u_imp
        self.kas = kas
        self.fuerzas=fuerzas
        self.xgrande=xgrande
        self.temperatura=temperatura


class Caso:
    def __init__(self, dominio, ccpunto):
        self.dominio= dominio
        self.ccpunto= ccpunto
        self.calculado=False
    


class Zona:
    '''
    Boundary 'zone' integrals are done here in 1D.
    Normalized nodes are at psi= 0, 0.5, 1.
    Coef of their functions are in Hzi(). They're used only there.
    '''
    
    # puntos de Gauss y sus pesos (3ptos en 1D, 0-->1)
    a=np.sqrt(3./5)
    x_Gaus=[(-a+1)/2, 0.5, (a+1)/2]
    w_Gaus=[2.5/9, 4./9, 2.5/9]


    def __init__(self, xi, yi):  # listas xi[] & yi[] de 3 elementos cada
        self.xi=xi
        self.yi=yi
    
    def Hzi(self,psi):
        coefHz= np.array([[ 1., -3., 2.], [ 0., 4., -4.], [ 0., -1., 2.]])
            # seria mas optimo pero menos claro asignar coefHz desde fuera
        a= np.matmul( coefHz, np.array([1, psi, psi*psi])  )
        return(a)

    def Hzi_psi(self,psi):   # las derivadas   d H / d psi
        a= np.array([-3.+4.*psi, 4.-8.*psi, -1.+4.*psi])
        return (a)

    def dimexy(self,psi):
        x= np.matmul(self.Hzi(psi), self.xi)
        y= np.matmul(self.Hzi(psi), self.yi)
        return(x,y)


    def n_ext(self, psi):     # vector normal exterior & vector tg
        a= np.matmul( self.xi, self.Hzi_psi(psi) )
        b= np.matmul( self.yi, self.Hzi_psi(psi) )
        c= np.hypot(a,b)
        t= np.array([a, b]) /c
        n= np.array([t[1], -t[0]])
        return(n,t)
    
    def pinta_zona(self, colorin='b', tipolin='-', ancholin=2.4, desplacin=0.,
                   alfalin=0.3, npl=17, todoL=False):
        # desplacin es un despl cte (pequeño) para marcar la zona en el dibujo
        # si se quiere dibujar una deformada, suministrar los nodos ya movidos
        # npl= num de ptos, la l de lado es por parecido a pinta_elem
        x,y=[],[]
        psi_ini, psi_fin= (0., 1.) if todoL else (0.02, 0.98)
        for psi in np.linspace(psi_ini, psi_fin, npl):  # llega o no a los extremos
            a,b= self.dimexy(psi)
            if desplacin:
                n,t= self.n_ext(psi)
                n *= desplacin
                a += n[0]
                b += n[1]
            x.append(a)
            y.append(b)
        plt.plot(x,y, tipolin, linewidth=ancholin, color=colorin, alpha=alfalin)
        return()

    

class ToolTip(object):
    # para los cuadradillos informativos emergentes
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", relief=SOLID, borderwidth=1,
                      font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()



def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def rellena_default():
    #  rellena como si se hiciese a mano. Todo es texto por tanto.
    global tplana_strv # este es StringVar()

    # DOMINIO
    
    nnodos=6
    texto= ''' 1    -9.0    0.0
 2    -4.0    2.5
 3     0.0    3.5
 4     0.0    4.25
 5     0.0    5.0
 6    -5.0    3.5'''
    wgt_nodos.insert('1.0', texto)

    nelems=1
        #  ielem, mat,   nodos  
    texto= '1   0    1 2 3 4 5 6'
    wgt_elems.insert('1.0', texto)
    
    texto= '0   1000.   0.2   1.e-3'
    wgt_ctes.insert('1.0', texto)

    tplana_strv.set('1')


    # SUSTENTACION

    texto= ' xy   1    0.0    0.0'
    wgt_nodos_imp.insert('1.0',texto)

    texto= 'nt   3 4 5   0.0  0.0  0.0    free  free  free'
    wgt_z_i.insert('1.0', texto)


    # CARGAS
    texto='nt  5 6 1    -10. -10. -10.     0. 0. 0.'
    wgt_xr.insert('1.0', texto)

        



def salida_prolija(ventana, icaso):
    # imprime datos de comprobacion y resultados si los hay
    
    global casos
    caso = casos[icaso]
    
    nn=len(caso.dominio.nodos)
    if nn > 400:
        texto='(the mesh for the requested output has {:} nodes)'.format(nn)
        confirmacion= messagebox.askyesno(parent=ventana,
            message='A big text file is going to be generated' , icon='question',
            detail=texto, title='Are you sure?', default='no')
        if not confirmacion: return()
            

    
    f=open('prolijo.out','w')
    
    print('\n', '*'*30, '\nOUTPUT for TESTING PURPOSES', file=f)
    print('\nNode   x         y', file=f)
    for inodo, nodo in caso.dominio.nodos.items():
        print('{:3d}  {:8.4f}  {:8.4f}'.format(inodo, nodo.x, nodo.y), file=f)
    print('\nElems:', file=f)

    for iel, el in caso.dominio.elems.items():
        print(iel, el.mat, el.nodse, file=f)
    print('\nCtes:', file=f)

    for i, lista in caso.dominio.ctes.items():
        print(i, lista, file=f)
    print('\nTP ?: ', caso.dominio.tplana, file=f)
    
    print('\nNodes with prescribed u:', file=f)
    for i,kk in caso.ccpunto.u_imp.items():
        print(i, kk, file=f)

    print('\nPuntual springs:', file=f)
    for i,kk in caso.ccpunto.kas.items():
        print(i, kk, file=f)

    print('\nZones with prescribed u (iel, cczona):', file=f)
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]== 'zu_imp':
                print(iel, ccz, file=f)

    print('\nSpring zones (iel, cczona):', file=f)
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]== 'zkas':
                print(iel, ccz, file=f)

    print('\nDistributed load zones (iel, cczona):', file=f)
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]== 'xraya':
                print(iel, ccz, file=f)
    print('\nVolume loading:', file=f)
    print(filas_xgrande[0].get(), file=f)
    print(filas_xgrande[1].get(), file=f)

    print('\nTemperature:', file=f)
    print(filas_temperatura[0].get(), file=f)
    
    print('\nPuntual forces:', file=f)
    for i,kk in caso.ccpunto.fuerzas.items():
        print(i, kk, file=f)

    if not caso.calculado: return()
    
    print('\n', '*'*60, '\nRESULTS\n', file=f)
    print('nodo    ux          uy         s_xx        s_yy          s_xy          s_vm',
            file=f)
    for inodo, nodo in caso.dominio.nodos.items():
        texto='{:}  {:10g}  {:10g}  {:10g}  {:10g}  {:10g}  {:10g}'.format(
            inodo, nodo.ux, nodo.uy, 
            nodo.sigma[0], nodo.sigma[1], nodo.sigma[2], nodo.sigma[3])
        print(texto,file=f)
    
    f.close()
    
    texto='Text listing correctly generated.'
    messagebox.showinfo(parent=ventana,message=texto,title='All fine')
    return()





def leer_gui():
    # Obtiene los datos del gui y crea el caso base.
    global casos
    free='free'
    
    def lee_widget(w):
        # devuelve las lineas[] del w=tk.text() como elementos de texto
        # prefiero leer por lineas para tener un poco mas de control de errores
        lineas, linea = [], 'hola'
        ilinea = 1
        while len(linea)>2:
            a= str(ilinea)+'.0'
            b= a+ 'lineend'
            linea=w.get(a, b).split()
            lineas.append(linea)
            ilinea += 1
        return(lineas[0:-1])
    

    ##################################
    # obtener datos de frame_dominio #
    ##################################
    
    nodos={}
    lineas_txt=lee_widget(wgt_nodos)
    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 3:
            try:
                inodo=int(linea[0])
                x=float(linea[1])
                y=float(linea[2])
            except:
                texto= f'Error in nodos[]: Line {ipos+1}.'
                texto+=' Pone: '+ str(linea)[1:-1].replace(',', '')
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
        else:
            texto= f'Required 3 data items for a node. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)

        n0= Nodo(x,y)
        nodos[inodo]=n0


    elems={}
    lineas_txt=lee_widget(wgt_elems)
    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 8:
            try:
                ielem= int(linea[0])
                mat= int(linea[1])
                nodse=[]
                for j in range(6):
                    nodse.append(int(linea[j+2]))
            except:
                texto= f'Error in elems[]: Line {ipos+1}.'
                texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
        else:
            texto= f'Required 8 data items for an elem. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)

        e0= Elem(mat=mat, nodse=nodse)
        elems[ielem]=e0


    if len(nodos)==0 or len(elems)==0:
        info_strv.set('BAD: There is no data to create a mesh.')
        casos=[]
        return(1)


    ctes={}  # antes era [], es mejor así, coherente y facilita columna "nº"
    lineas_txt=lee_widget(wgt_ctes)
    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 4:
            try:
                imat= int(linea[0])
                young= float(linea[1])
                pois= float(linea[2])
                alfaT= float(linea[3])
            except:
                texto= f'Error in materials: Line {ipos+1}.'
                texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
        else:
            texto= f'Required 4 data items to describe a material. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)

        ctes[imat]=[young, pois, alfaT]
    
    tplana= int(tplana_strv.get())

    dominio=Dominio(nodos, elems, ctes, tplana)
    dominio.pon_basicos()
    dominio.pon_extremos()
    dominio.pon_vecinos()
    dominio.pon_contornos()
    dominio.pon_pseudo_n_ext() # al completo, es para el caso base


    #######################################
    # obtener datos de frame_sustentacion #
    #######################################
    
    u_imp = {}
    lineas_txt=lee_widget(wgt_nodos_imp)  # apoyos punctuales
    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 4:
            if linea[0]=='nt':
                texto= f'Error in prescribed u: Line {ipos+1}. '
                texto+='"nt" is not valid for a single node, but'
                texto+='you can specify a tilt angle.'
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
            elif linea[0]=='xy':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except (TypeError, ValueError):
                    texto= f'Error in prescribed u: Line {ipos+1}. '
                    texto+='The "dir field is not understood.'
                    texto+='\n             '+ str(sys.exc_info()[0])
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)

            try:
                inodo= int(linea[1])
            except:
                texto= f'Error in prescribed u: Line {ipos+1}. '
                texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
                
            if linea[2]==free and linea[3]== free:
                texto= f'Error in prescribed u: Line {ipos+1}. '
                texto+='At least one displacement component'
                texto+='must have a value (not "free").'
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
                
            try:
                u1= float(linea[2])
            except (TypeError, ValueError):
                u1=free
            try:
                u2= float(linea[3])
            except  (TypeError, ValueError):
                u2=free

        else:
            texto= f'Required 4 data items for a prescribed u. Line {ipos+1}. '
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
    
        u_imp[inodo]=[nt,u1,u2]


    kas={}
    lineas_txt=lee_widget(wgt_r_p)  # resortes punctuales
    for ipos,linea in enumerate(lineas_txt):
        if len(linea)==4:
            if linea[0]=='nt':
                texto= f'Error in punctual k: Line {ipos+1}. '
                texto+='"nt" is not valid for a single node, but'
                texto+='you can specify a tilt angle.'
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
            elif linea[0]=='xy':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except (TypeError, ValueError):
                    texto= f'Error in punctual k: Line {ipos+1}. '
                    texto+='"dir" field is not understood.'
                    texto+='\n             '+ str(sys.exc_info()[0])
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)

            try:
                inodo= int(linea[1])
                k1= float(linea[2])
                k2= float(linea[3])
            except (TypeError, ValueError):
                texto= f'Error in punctual k: Line {ipos+1}. '
                texto+='Please check the data.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)

        else:
            texto= f'Required 4 data items for a punctual spring. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)

        kas[inodo]=[nt,k1,k2]


    # recordar: las cczona se guardan en los elems[] como listas:
    for el in dominio.elems.values(): el.cczona=[]


    lineas_txt=lee_widget(wgt_z_i)  # zonas de u impuesto

    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 10:
            
            if linea[0]=='xy' or linea[0]=='nt':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except:
                    texto= f'Error in prescribed u zone: Line {ipos+1}. '
                    texto+='"dir" field not understood.'
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)
            
            try:
                ns= [ int(linea[1]), int(linea[2]), int(linea[3]) ]
            except:
                texto= f'Error in prescribed u zone: Line {ipos+1}. '
                texto+='Please check the data.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
                    
            if linea[4]==free or linea[5]==free or linea[6]==free:
                u1s=[free,free,free]
            else:
                try:
                    u1s=[ float(linea[4]), float(linea[5]), float(linea[6]) ]
                except:
                    texto= f'Error in prescribed u zone: Line {ipos+1}. '
                    texto+='Please ckeck the data.'
                    texto+='\n             '+ str(sys.exc_info()[0])
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)

            if linea[7]==free or linea[8]==free or linea[9]==free:
                u2s=[free,free,free]
            else:
                try:
                    u2s=[ float(linea[7]), float(linea[8]), float(linea[9]) ]
                except:
                    texto= f'Error in prescribed u zone: Linea {ipos+1}. '
                    texto+='Please check the data.'
                    texto+='\n             '+ str(sys.exc_info()[0])
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)

            # a ver de que elemento es y guardarlo ahi:
            for iel, el in dominio.elems.items():
                lado_el=-1
                if el.nodse[0:3]==ns:
                    lado_el=0
                elif el.nodse[2:5]==ns:
                    lado_el=1
                elif [el.nodse[4],el.nodse[5],el.nodse[0]]==ns:
                    lado_el=2
                if lado_el != -1:
                    a=[nt]+ns+u1s+u2s+['zu_imp']+[lado_el]
                    el.cczona.append(a)
                    break  # solo puede ser de un elemento

        else:
            texto= f'Required 10 data items for a prescribed u zone. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)



    lineas_txt=lee_widget(wgt_r_s)  # zonas de resortes del contorno

    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 10:
            if linea[0]=='xy' or linea[0]=='nt':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except:
                    texto= f'Error in spring boundary zone: Line {ipos+1}. '
                    texto+='"dir" field not understood.'
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)
            
            try:
                ns= [ int(linea[1]), int(linea[2]), int(linea[3]) ]
                k1s=[ float(linea[4]), float(linea[5]), float(linea[6]) ]
                k2s=[ float(linea[7]), float(linea[8]), float(linea[9]) ]
            except:
                texto= f'Error in spring boundary zone: Line {ipos+1}. '
                texto+='Please check the data.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
        else:
            texto= f'Required 10 data items for a spring zone: Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        
        # a ver de que elemento es y guardarlo ahi:
        for iel, el in dominio.elems.items():
            lado_el=-1
            if el.nodse[0:3]==ns:
                lado_el=0
            elif el.nodse[2:5]==ns:
                lado_el=1
            elif [el.nodse[4],el.nodse[5],el.nodse[0]]==ns:
                lado_el=2
            if lado_el != -1:
                a=[nt]+ns+k1s+k2s+['zkas']+[lado_el]
                el.cczona.append(a)
                break  # solo puede ser de un elemento


    #################################
    # obtener datos de frame_cargas #
    #################################

    lineas_txt=lee_widget(wgt_xr)  # zonas de carga distribuida xraya

    for ipos,linea in enumerate(lineas_txt):
        if len(linea) == 10:
            if linea[0]=='xy' or linea[0]=='nt':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except:
                    texto= f'Error in distributed load zone: Line {ipos+1}. '
                    texto+='"dir" field is not understood.'
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)
            
            try:
                ns= [ int(linea[1]), int(linea[2]), int(linea[3]) ]
                x1s=[ float(linea[4]), float(linea[5]), float(linea[6]) ]
                x2s=[ float(linea[7]), float(linea[8]), float(linea[9]) ]
            except:
                texto= f'Error in distributed load zone: Line {ipos+1}. '
                texto+='Please check the data.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
        else:
            texto= f'Required 10 data items for a distributed load zone: Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        
        # a ver de que elemento es y guardarlo ahi:
        for iel, el in dominio.elems.items():
            lado_el=-1
            if el.nodse[0:3]==ns:
                lado_el=0
            elif el.nodse[2:5]==ns:
                lado_el=1
            elif [el.nodse[4],el.nodse[5],el.nodse[0]]==ns:
                lado_el=2
            if lado_el != -1:
                a=[nt]+ns+x1s+x2s+['xraya']+[lado_el]
                el.cczona.append(a)
                break  # solo puede ser de un elemento



    xgrande=[]  # fuerzas de volumen & temperatura
    if filas_xgrande[0].get() != '':
        xgrande.append(filas_xgrande[0].get())
        xgrande.append(filas_xgrande[1].get())
        if valida_expr(xgrande): return(1)

    temperatura=[]
    if filas_temperatura[0].get() != '':
        temperatura.append( filas_temperatura[0].get())
        if valida_expr(temperatura): return(1)



    fuerzas={}  # fuerzas puntuales
    lineas_txt=lee_widget(wgt_fuerzas)
    for ipos,linea in enumerate(lineas_txt):
        if len(linea)==4:
            if linea[0]=='nt':
                texto= f'Error in punctual force: Line {ipos+1}. '
                texto+='"nt" is not valid in a sigle node, but'
                texto+='you can specify a tilt angle.'
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
            elif linea[0]=='xy':
                nt=linea[0]
            else:
                try:
                    nt=float(linea[0])
                except (TypeError, ValueError):
                    texto= f'Error in punctual force: Line {ipos+1}. '
                    texto+='"dir" field not undesrtood.'
                    texto+='\n             '+ str(sys.exc_info()[0])
                    messagebox.showinfo(message=texto,title='BAD',parent=v0)
                    return(1)

            try:
                inodo= int(linea[1])
                F1= float(linea[2])
                F2= float(linea[3])
            except (TypeError, ValueError):
                texto= f'Error in punctual force: Line {ipos+1}. '
                texto+='Please ckeck the data.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)

        else:
            texto= f'Required 4 data items for a punctual force. Line {ipos+1}.'
            texto+=' It states: '+ str(linea)[1:-1].replace(',', '')
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)

        fuerzas[inodo]=[nt,F1,F2]


    ccpunto= CCpunto(u_imp, kas, fuerzas, xgrande, temperatura)
    
    caso= Caso(dominio, ccpunto)
    casos=[caso]  # leer_gui() resetea los casos
    
    info_strv.set('Mesh has been generated from data') # se sobreescribe si no se llamo desde boton
    ncasos_strv.set('1') 

    return(caso)



def valida_expr (ff):
    # entrada= una lista de funciones de x,y, & terminos matematicos
    bien=' 0123456789+-*/.exy()'
    for ffi in ff:
        a=ffi
        a.strip()
        esta_N=a
        a= a.replace('exp','').replace('sin','').replace('cos','').replace('log','')
        for b in bien:
            a=a.replace(b,'')
        if len(a): # no debería quedar nada
            texto= 'The expression: \n'
            texto+= esta_N +'\nis not a valid function.'
            texto+='\nPlease check it out.'
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        else:
            try:
                a=esta_N
                x,y= 2.2, 3.3
                kk=eval(esta_N)
            except (TypeError, ValueError, SyntaxError, NameError, AttributeError):
                texto= 'The following function cannot be evaluated:\n'
                texto += esta_N or '   - (blank space) -'
                texto+='\nPLease check the syntaxis.'
                texto+='\n             '+ str(sys.exc_info()[0])
                messagebox.showinfo(message=texto,title='BAD',parent=v0)
                return(1)
    return(0)




def verXT():
    # visualiza contornos para temperatura y quiver para xraya. Ya tiene que 
    # haber pasado por leer_gui -> valida_expr(). Sobran mas comprobaciones.
    global casos # mejor que releer y recomprobar
    texto= 'You must load the case ("read gui" button) before generating'
    texto+='this visualization.'
    if len (casos)==0:
        messagebox.showinfo(message=texto,title='BAD',parent=v0)
        return(1)

      
    xe,ye = [], []  # puntos para evaluar. Seran los nodos, y los
                    # centros de los sub-triangulos
    for n in casos[0].dominio.nodos.values():
        xe.append(n.x)
        ye.append(n.y)
    for el in casos[0].dominio.elems.values():
        ptos_ = [[0, 0.58],[0, 1.15],[0.5, 1.44],[-0.5, 1.44]]
        for p in ptos_: # centros de sub triangulos
            xi, yi = el.dimexy( p[0], p[1] )
            xe.append(xi)
            ye.append(yi)
    if len(casos[0].dominio.nodos)<13:
        ptos_ = [[-0.25, 0.43,],[0.25, 0.43],[-0.75, 1.3],[0.75, 1.3],
                 [-0.5, 1.73],[0.5, 1.73]]
        for p in ptos_: # cuartos de lados
            xi, yi = el.dimexy( p[0], p[1] )
            xe.append(xi)
            ye.append(yi)


    if len(casos[0].ccpunto.xgrande):

        u,v, max_xgrande = [], [], 0.
        for i in range(len(xe)): # len(xe) es el num de ptos
            x,y = xe[i], ye[i]
            a,b= eval(casos[0].ccpunto.xgrande[0]), eval(casos[0].ccpunto.xgrande[1])
            u.append(a)
            v.append(b)
            c= np.hypot(a,b)
            if c > max_xgrande: max_xgrande = c
        
        plt.close(fig='Volume force')
        plt.figure('Volume force')
        plt.quiver(xe,ye,u,v, units='dots', width=2, color='r')
        texto= 'Max value= {:.2g}'.format(max_xgrande)
        plt.annotate(texto, xy=(0.04,0.94), xycoords='axes fraction', 
             fontsize=9, color='r', fontweight='roman')
        
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')


    if len(casos[0].ccpunto.temperatura):
        plt.close(fig='Temperature')
        plt.figure('Temperature')
        xe = np.linspace(casos[0].dominio.extremosx[0], casos[0].dominio.extremosx[1], 9)
        ye = np.linspace(casos[0].dominio.extremosy[0], casos[0].dominio.extremosy[1], 9)
        x,y = np.meshgrid(xe, ye)
        t= eval(casos[0].ccpunto.temperatura[0])
        rellenos= plt.contourf(xe, ye, t, cmap='coolwarm', alpha=0.7)
        plt.colorbar(rellenos, label='temperature')


        #for i in range(len(xe)):
        #    x,y = xe[i], ye[i]
        #    t.append(eval(casos[0].ccpunto.temperatura[0]))
        #plt.scatter(xe,ye, s=900, c=t, cmap='coolwarm', alpha=0.5, edgecolors='none')
        #rellenos= plt.tricontourf(xe,ye, t, cmap='coolwarm', alpha=0.7)
        #plt.colorbar(rellenos, label='temperatura')

        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')

    plt.show()




def a_guardar():
    
    leer_gui()
    caso=casos[0] # guardara lo que hay en el gui ( -> ha reseteado los casos)
    if caso == 1:
        texto= 'Data contains errors.'
        texto+='Saving is not possible in this situation.'
        texto+='\n             '+ str(sys.exc_info()[0])
        messagebox.showinfo(message=texto,title='BAD',parent=v0)
        return(1)
    
    nfcompleto = filedialog.asksaveasfilename(parent=v0, title='Save problem')
    if not nfcompleto:
        return(1)

    n_f = path.basename(nfcompleto)
    v0.title('jMef v0.6 - '+n_f)
    
    nnodos, nelems= len(caso.dominio.nodos), len(caso.dominio.elems)

    f=open(nfcompleto,'w')
    
    print('Identification: ', nfcompleto,file=f)
    print('\nNum_of_nodes     Num_of_elements', file=f)
    print(' {:6d}  {:15d}'.format(nnodos, nelems),file=f)
    print('-'*60, file=f)
    
    print('iNode    coor_x      coor_y', file=f)
    for i,n in caso.dominio.nodos.items():
        print('{:3d}  {:10.5f}  {:10.5f}'.format(i, n.x, n.y),file=f)

    print('\niElem  mat      nodes', file=f)
    for i,e in caso.dominio.elems.items():
        s = '{:}    {:}  '.format(i, e.mat)
        for n in e.nodse:
            s += ' ' + str(n)
        print(s, file=f)
    
    print('\nMaterials:   ',len(caso.dominio.ctes), file=f)
    print(' nº    E       nu     alfa_temp', file=f)
    for i,c in caso.dominio.ctes.items():
        print(' {:}   {:}   {:}    {:}'.format(i, c[0], c[1], c[2]), file=f)
    
    c='TP' if caso.dominio.tplana==1 else 'DP'
    print('\nTP/DP?:  ', c, file=f)
    

    
    # En dir (eccion) puede haber:
    # un float -> el angulo de la coor1 con el ejex
    # str 'nt' -> la coor1 es n_ext
    # xy ó ''  -> la coor 1 es x (escribimos 'xy' explicito)
    # En los u_1, u_2 puede haber 'free' en ciertas componentes
    
    print('-'*60, file=f)
    print('Nodes with prescribed displacement: ', len(caso.ccpunto.u_imp), file=f)
    print(' dir  Node     u_1     u_2', file=f)
    for n,a in caso.ccpunto.u_imp.items():
        s=' xy ' if a[0]=='' else str(a[0])
        s += ' {:d}      {:}     {:}'.format(n, a[1], a[2])
        print(s, file=f)
    
    print('\nPunctual springs in nodes: ', len(caso.ccpunto.kas), file=f)
    print(' dir  Node     k_1     k_2', file=f)
    for n,a in caso.ccpunto.kas.items():
        s=' xy ' if a[0]=='' else str(a[0])
        s += '  {:d}      {:}      {:}'.format(n, a[1], a[2])
        print(s, file=f)
    
    lista_zu_imp, lista_zkas, lista_xraya = [],[],[]
    for ie,e in caso.dominio.elems.items():
        for a in e.cczona:
            if a[-2]=='zu_imp': 
                lista_zu_imp.append(a[0:11])
            elif a[-2]=='zkas':
                lista_zkas.append(a[0:11])
            elif a[-2]=='xraya':
                lista_xraya.append(a[0:11])
            else:
                print(f'Found invalid boundary condition in elem {ie}.')
                return(-1)
    
    print('\nPrescribed displacement zones: ',len(lista_zu_imp),file=f)
    print(' dir  nA  nB  nC    u1(A)  u1(B)  u1(C)    u2(A)  u2(B)  u2(C)', file=f)
    for a in lista_zu_imp:
        s=' xy ' if a[0]=='' else str(a[0])
        s += '  {:d} {:d} {:d} '.format(a[1], a[2], a[3])
        s += ' {:}  {:}  {:}    {:}  {:}  {:}'.format(
                                            a[4], a[5], a[6], a[7], a[8], a[9])
        print(s, file=f)
        
    print('\nElastic foundation boundary zones: ', len(lista_zkas), file=f)
    print(' dir  nA  nB  nC    k1/L  k1/L  k1/L    k2/L  k2/L  k2/L', file=f)
    for a in lista_zkas:
        s=' xy ' if a[0]=='' else str(a[0])
        s += '  {:d} {:d} {:d} '.format(a[1], a[2], a[3])
        s += ' {:}  {:}  {:}    {:}  {:}  {:}'.format(
                                            a[4], a[5], a[6], a[7], a[8], a[9])
        print(s, file=f)
    
    print('-'*60, file=f)
    print('Boundary load zones: ', len(lista_xraya), file=f)
    print(' dir  nA  nB  nC    X1/L  X1/L  X1/L    X2/L  X2/L  X2/L', file=f)
    for a in lista_xraya:
        s=' xy ' if a[0]=='' else str(a[0])
        s += '  {:d}  {:d}  {:d} '.format(a[1], a[2], a[3])
        s += ' {:}  {:}  {:}    {:}  {:}  {:}'.format(
                                            a[4], a[5], a[6], a[7], a[8], a[9])
        print(s, file=f)
    
    print('\nVolume forces: ', len(caso.ccpunto.xgrande)//2, file=f)
    for a in caso.ccpunto.xgrande:
        print('  ' + a[0], file=f)
        print('  ' + a[1], file=f)
    
    print('\nTemperature field: ', len(caso.ccpunto.temperatura), file=f)
    for a in caso.ccpunto.temperatura:
        print('  ' + a[0], file=f)
    
    print('\nPunctual forces in nodes: ', len(caso.ccpunto.fuerzas), file=f)
    print(' dir  Nodo    F_1      F_2', file=f)
    for n,a in caso.ccpunto.fuerzas.items():
        s=' xy ' if a[0]=='' else str(a[0])
        s += '    {:d}   {:}   {:}'.format(n, a[1], a[2])
        print(s, file=f)

    print('# ___end___\n\n', file=f)
    f.close()
    
    info_strv.set('Problem saved (using mesh 0).')
    ncasos_strv.set('1')
    texto= 'File is saved in\n'
    texto+= nfcompleto
    messagebox.showinfo(message=texto,title='Done',parent=v0)



def especial():
    # hacer de un caso superior el caso 0 // ver el problema sobre una malla superior

    global casos

    def caso_a_gui(icaso):
        global casos
        caso = casos[icaso]
        a_borrar()
        
        # llenar dominio
        
        texto=''
        for i, n in caso.dominio.nodos.items():
            texto += '{:3d}  {:10.5f}  {:10.5f}\n'.format(i, n.x, n.y)
        wgt_nodos.insert('1.0', texto)
        
        texto=''
        for iel, el in caso.dominio.elems.items():
            s = '{:}    {:}'.format(iel, el.mat)
            for n in el.nodse:
                s += ' ' + str(n)
            s +='\n'
            texto += s
        wgt_elems.insert('1.0', texto)

        texto=''
        for i, c in caso.dominio.ctes.items():
            texto += ' {:}   {:}   {:}    {:}\n'.format(i, c[0], c[1], c[2])
        wgt_ctes.insert('1.0', texto)

        tplana_strv.set(str(caso.dominio.tplana))
        
        # llenar sustentacion
        
        texto=''
        for n,a in caso.ccpunto.u_imp.items():
            s=' xy ' if a[0]=='' else str(a[0])
            s += ' {:d}   {:}   {:}\n'.format(n, a[1], a[2])
            texto += s
        wgt_nodos_imp.insert('1.0', texto)
    
        texto=''
        for n,a in caso.ccpunto.kas.items():
            s=' xy ' if a[0]=='' else str(a[0])
            s += ' {:d}   {:}   {:}\n'.format(n, a[1], a[2])
            texto += s
        wgt_r_p.insert('1.0', texto)
        
        texto_zu, texto_zk, texto_xr = '', '', ''
        for ie,e in caso.dominio.elems.items():
            for a in e.cczona:
                if a[-2]=='zu_imp':
                    s='{:}   {:} {:} {:}   {:}  {:}  {:}   {:}  {:}  {:}\n'.format(
                            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9])
                    texto_zu += s
                elif a[-2]=='zkas':
                    s='{:}   {:} {:} {:}   {:}  {:}  {:}   {:}  {:}  {:}\n'.format(
                            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9])
                    texto_zk += s
                elif a[-2]=='xraya':
                    s='{:}   {:} {:} {:}   {:}  {:}  {:}   {:}  {:}  {:}\n'.format(
                            a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9])
                    texto_xr += s
        wgt_z_i.insert('1.0', texto_zu)
        wgt_r_s.insert('1.0', texto_zk)
        wgt_xr.insert('1.0', texto_xr)


        # llenar cargas (salvo xraya que acabo de poner)
        
        try:
            filas_xgrande[0].insert(0, caso.ccpunto.xgrande[0])
            filas_xgrande[1].insert(0, caso.ccpunto.xgrande[1])
        except IndexError:
            pass
            
        try:
            filas_temperatura[0].insert(0, caso.ccpunto.temperatura[0])
        except IndexError:
            pass
        
        texto = ''
        for n,a in caso.ccpunto.fuerzas.items():
            s=' xy ' if a[0]=='' else str(a[0])
            s += ' {:d}   {:}   {:}\n'.format(n, a[1], a[2])
            texto += s
        wgt_fuerzas.insert('1.0', texto)


    
    def haz_especial():
        global casos
        i= int(i_di_verconcc.get())
        j= int(cambiar_caso0_strv.get())
        k= int(salida_prolija_strv.get())
        if j and k:
            texto='Both options are incompatible.'
            messagebox.showinfo(message=texto,title='BAD',parent=v2,
                detail='Please choose one of them now and eventually revisit this window.')
            return()
        
        if i and not (j or k):
            pinta_caso(i)
            info_strv.set('Boundary conditions have been displayed on a refined mesh.')
        if i and j:
            casos=[casos[i]]
            casos[0].calculado = False  # lo recalculas, si eso
            ncasos_strv.set('1')
            info_strv.set('A new "base case" has been established by using a refined mesh.')
            caso_a_gui(0)
        if i and k:
            salida_prolija(v2,i)
        
        v2.destroy()
        return()

    if len(casos)==0:
        texto='No data to operate on.'
        messagebox.showinfo(message=texto,title='BAD',parent=v2,
            detail='Please load or type the data for a case.')
        return(1)

    v2=Toplevel(v0)
    v2.title('Special actions')
    
    texto='''These actions will not be in general required to understand the solid behaviour or deciding on design aspects. It is advised to start with a better "base case" (mesh 0) in first place, and to avoid these actions whenever possible.'''
    
    tcaja = Text(v2, width=50, height=6,wrap='word', font=('Sans',11),
            background='#EDECEB', foreground='green', borderwidth=0, padx=10, pady=12)
    tcaja.grid(column=0, row=0, columnspan=4, padx=8, sticky=(N,W,E,S))
    tcaja.insert('1.0',texto)
    tcaja['state'] = 'disabled'
    
    texto='''To see the boundary conditions on a refined mesh (with no effect on the analysis), please pick the desired mesh and hit "proceed".'''
    tcaja = Text(v2, width=30, height=5, wrap='word', font=('Sans',10), 
            background='#EDECEB', borderwidth=0, padx=10, pady=12)
    tcaja.grid(column=3, row=1, rowspan=len(casos), padx=8)
    tcaja.insert('1.0',texto)
    tcaja['state'] = 'disabled'
    
    botonicos, i_di_verconcc = [], StringVar(value='0')
    for i in range(len(casos)):
        nnodos= len(casos[i].dominio.nodos)
        texto = f'Mesh {i} ({nnodos} nodes)'
        c='blue' if casos[i].calculado else 'black'
        ttk.Label(v2, text=texto,background='#EDECEB', foreground=c, font=('Sans',10)).grid(
                    row=i+1, column=1, padx=8, pady=3, sticky='w')
        
        botonico=Radiobutton(v2, variable=i_di_verconcc, value=str(i),background='#EDECEB')
        botonico.grid(row=i+1, column=0, padx=4, pady=4, sticky='w') 
        botonicos.append(botonico)
    i_di_verconcc.set('0')

    ttk.Separator(v2,orient=VERTICAL).grid(row=1, column=2, rowspan=len(casos),
                 pady=10, sticky='ns')
    
    ttk.Separator(v2,orient=HORIZONTAL).grid(row=len(casos)+2, column=0, columnspan=4,
                 padx=20, pady=6, sticky='ew')
    
    texto='''Please check this option if you want to make a new "base case" from the choosen mesh. Typically, if you provided a mesh with only one or two elements and you find the output scarce, choose mesh 1 (or at most mesh 2) and check this box.'''
    tcaja = Text(v2, width=30, height=9, wrap='word', font=('Sans',10),
            background='#EDECEB', borderwidth=0, padx=10, pady=12)
    tcaja.grid(column=3, row=len(casos)+3, padx=8)
    tcaja.insert('1.0',texto)
    tcaja['state'] = 'disabled'
    
    ttk.Separator(v2,orient=VERTICAL).grid(row=len(casos)+3, column=2,
                 pady=10, sticky='ns')
    
    cambiar_caso0_strv = StringVar()
    boton_cambiar_caso0 = ttk.Checkbutton(v2, text='be "base case"', variable=
        cambiar_caso0_strv, onvalue='1', offvalue='0', style='jc.TCheckbutton')
    boton_cambiar_caso0.grid(row=len(casos)+3, column=0, columnspan=2, padx=6)
    cambiar_caso0_strv.set('0')
    
    ttk.Separator(v2,orient=HORIZONTAL).grid(row=len(casos)+4, column=0, columnspan=4,
                 padx=20, pady=6, sticky='ew')
    
    texto='''This option produces a long (tedious at human sight) listing, matching the mesh specified above. The output will be stored in a file named "prolijo.out" in the working directory. If you didn't yet calculate for that mesh, only data will be listed. The already calculated meshes are shown in blue for your convenience.'''
    tcaja = Text(v2, width=30, height=11, wrap='word', font=('Sans',10),
        background='#EDECEB', borderwidth=0, padx=10, pady=12)
    tcaja.grid(column=3, row=len(casos)+5, padx=8)
    tcaja.insert('1.0',texto)
    tcaja['state'] = 'disabled'
    
    salida_prolija_strv= StringVar()
    boton_salida_prolija = ttk.Checkbutton(v2, text='text output', variable=
        salida_prolija_strv, onvalue='1', offvalue='0', style='jc.TCheckbutton')
    boton_salida_prolija.grid(row=len(casos)+5, column=0, columnspan=2, padx=6)
    salida_prolija_strv.set('0')
    
    ttk.Separator(v2,orient=VERTICAL).grid(row=len(casos)+5, column=2,
                 pady=10, sticky='ns')
    
    ttk.Separator(v2,orient=HORIZONTAL).grid(row=len(casos)+6, column=0, columnspan=4,
             padx=20, pady=6, sticky='ew')
    
    boton_haz_especial=ttk.Button(v2, text='proceder' , command= haz_especial)
    boton_haz_especial.grid(column=0, row=len(casos)+7, columnspan=4,padx=3, pady=5)
    
    v2.mainloop()




def a_limpiar(): # ventana de borrar por partes
    
    def hecholimpiar():
        
        a_borrar(dom=lista_bool[0].get(), 
                 sus=lista_bool[1].get(), 
                 car=lista_bool[2].get() )
        v3.destroy()


    v3=Toplevel(v0)
    v3.title('Field cleaning')
    v3.geometry('+300+200')

    lista_bool=[BooleanVar(), BooleanVar(), BooleanVar()]
            # son variables de los botones
    
    a= 'Please tick the input fields\n'
    a+=' you want to set as blank:'
    ttk.Label(v3, text=a, background='#EDECEB').grid(
            row=0, column=0, columnspan=2)
            
    ttk.Checkbutton(v3, variable=lista_bool[0], style='jc.TCheckbutton').grid(
            row=1, column=0)
    ttk.Label(v3, text='Domain', background='#EDECEB').grid(
            row=1, column=1, sticky='w')

    ttk.Checkbutton(v3, variable=lista_bool[1], style='jc.TCheckbutton').grid(
            row=2, column=0)
    ttk.Label(v3, text='Bearing', background='#EDECEB').grid(
            row=2, column=1, sticky='w')
    
    ttk.Checkbutton(v3, variable=lista_bool[2], style='jc.TCheckbutton').grid(
            row=3, column=0)
    ttk.Label(v3, text='Loads', background='#EDECEB').grid(
            row=3, column=1, sticky='w')
    
    ttk.Button(v3, text='Proceed', command=hecholimpiar).grid(
            row=4, column=0, columnspan=2)

    for hijo in v3.winfo_children(): hijo.grid_configure(padx=6,
        pady=6)

    #v3.geometry('+410-70')
    v3.focus()
    v3.mainloop()


def a_borrar(dom=True, sus=True, car=True):
    # deja partes del gui en blanco, salvo factores de carga
    if dom:
        wgt_nodos.delete(1.0,END)
        wgt_elems.delete(1.0,END)
        wgt_ctes.delete(1.0,END)
        #esp_strv.set('1.0') # mejor no jugar con este
        tplana_strv.set('1')
    if sus:
        wgt_nodos_imp.delete(1.0,END)
        wgt_r_p.delete(1.0,END)
        wgt_z_i.delete(1.0,END)
        wgt_r_s.delete(1.0,END)
    if car:
        wgt_xr.delete(1.0,END)
        wgt_fuerzas.delete(1.0,END)
        filas_xgrande[0].delete(0,'end')
        filas_xgrande[1].delete(0,'end')
        filas_temperatura[0].delete(0,'end')


def a_salir():
    if messagebox.askokcancel(message='¿Do you want to close the program? ',
                detail='  Your cellphone will be formatted.', default='cancel',
                icon='question', title='Confirmation:',parent=v0) :
        for i in plt.get_fignums(): plt.close(i)
        
        try: v5.destroy() # la de personalizar graficos
        except: pass
        
        v0.destroy()
        try: # si es linux que cierre el terminal tambien
            kill(getppid(), signal.SIGHUP)        
        except:
            exit()
        # con extension .pyw no saca terminal.


def a_cargar():
    #  Lectura de fichero. Simplemente rellena el gui como a mano,
    #  y finalmente llama a leer_gui (que para eso esta)
    # no se si se necesitan todos los wigets txt en global ...
    global tplana_strv 
    global filas_xgrande, filas_temperatura
    global casos

    free='free'
    a_borrar()

    nfcompleto=filedialog.askopenfilename(parent=v0, title='Open file')
    if not nfcompleto: return()
    n_f= path.basename(nfcompleto)
    v0.title('jMef - '+n_f)

    f=open(nfcompleto, 'r')
    
        ### leer el dominio ###

    kk= f.readline()    # Titulo o lo que quieras
    kk= f.readline()    #linea en blanco
    kk= f.readline()    # texto Num nodos, Num elems
    kk= f.readline().split()
    nnodos, nelems = int(kk[0]), int(kk[1])
    
    kk= f.readline()    #linea en blanco
    kk= f.readline()    # texto iNodo, coor x, coor y
    for i in range(nnodos):
        kk=f.readline().strip()
        wgt_nodos.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline()    #linea en blanco
    kk= f.readline()       # texto iElem, mat, nodos del elemento
    for i in range(nelems):
        kk=f.readline().strip()
        wgt_elems.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline() 
    kk= f.readline()    # texto ' materiales:   xx
    n=int(kk.split()[-1])
    kk= f.readline()  # texto 'nº   E   nu    alfaT
    for i in range(n):
        kk=f.readline().strip()
        wgt_ctes.insert(str(i+1)+'.0' , kk+'\n')
        
    esp_strv.set('1.0') # por definicion, a no cambiar (es por recordatorio)
    
    kk=f.readline()
    kk=f.readline().split()  # texto TP/DP?
    if kk[-1]=='TP':
        tplana_strv.set('1') 
    else:
        tplana_strv.set('0') 

    ### leer la sustentacion ###

    kk= f.readline() 
    kk= f.readline()    # texto 'Nodos con desplazamiento impuesto:  xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto ' dir  Nodo         u_1         u_2'
    for i in range(n):
        kk=f.readline().strip()
        wgt_nodos_imp.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline() 
    kk= f.readline()    # texto 'Resortes puntuales en los nodos:   xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto ' dir  Nodo         k_1         k_2'
    for i in range(n):
        kk=f.readline().strip()
        wgt_r_p.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline() 
    kk= f.readline()    # Zonas con desplazamiento impuesto:   xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto  ' dir    Nodos     u_1     u_2'
    for i in range(n):
        kk=f.readline().strip()
        wgt_z_i.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline() 
    kk= f.readline()    # Zonas con apoyo elastico:   xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto  ' dir    Nodos     k1/L     k2/L'
    for i in range(n):
        kk=f.readline().strip()
        wgt_r_s.insert(str(i+1)+'.0' , kk+'\n')


    ### leer las cargas ###

    kk= f.readline() 
    kk= f.readline()    # texto 'Zonas con carga de contorno:   xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto  ' dir    Nodos     X_1     X_2'
    for i in range(n):
        kk=f.readline().strip()
        wgt_xr.insert(str(i+1)+'.0' , kk+'\n')

    kk= f.readline() 
    kk= f.readline()    # texto 'Fuerzas de volumen:   xx'
    n=int(kk.split()[-1]) # habra una o ninguna carga de V, en dos componentes
    if n:
        filas_xgrande[0].insert(f.readline().strip() )
        filas_xgrande[1].insert(f.readline().strip() )

    kk= f.readline() 
    kk= f.readline()    # texto 'Distribución de temperatura:   xx'
    n=int(kk.split()[-1])
    if n:
        filas_temperatura.insert(f.readline().strip() )

    kk= f.readline() 
    kk= f.readline()    # texto 'Cargas puntuales en nodos:   xx'
    n=int(kk.split()[-1])
    kk= f.readline()    # texto ' dir  Nodo         F_1         F_2'
    for i in range(n):
        kk=f.readline().strip()
        wgt_fuerzas.insert(str(i+1)+'.0' , kk+'\n')

    f.close()

    leer_gui() # resetea los casos[]
    info_strv.set('Data loaded into the user interface. A mesh has been generated.')
    ncasos_strv.set('1')
    return()



def cierraplots():
    global v5
    for i in plt.get_fignums(): plt.close(i)
    try:
        v5.destroy() # tambien la de personalizar graficos
    except:
        pass



def pinta_figurica(x,y, tipo_fig, n_ext):
    '''
    opciones del marker:
    marker='o', linestyle=':', markersize=15, color='darkgrey',
        markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue',
        markeredgecolor='brown', fillstyle='left'
    -> fillstyle={'full', 'left', 'right', 'bottom', 'top', 'none' }
    '''    

    def girar_path(path,giro):
        path=np.array(path)
        x= path[:,0]
        y= path[:,1]
        r= np.hypot(x,y)
        t= np.arctan2(y,x)+giro
        x= r*np.cos(t)
        y= r*np.sin(t)
        path_nuevo=np.array([x,y]).T
        return(path_nuevo)
    
    giro= np.arctan2(n_ext[1], n_ext[0])
    if tipo_fig=='muelle':
        el_path = girar_path([[0,0],[6,0],[10,10],[14,-10],[18,10],[22,-10],[26,0],
        [30,0],[40,12],[40,-12],[30,0]] , giro)
        plt.plot(x,y, marker=el_path, markersize=34, color='w', linestyle='None',
            markeredgecolor='b', markeredgewidth= 0.8, alpha=0.8)
    elif tipo_fig=='apoyof':
        plt.plot(x,y,marker='x',markersize=10, color='b', linestyle='None',
                linewidth=1.8, alpha=0.8)
    elif tipo_fig=='apoyom':
        el_path=[[0,0],[12,-8],[13,-6],[13.6,-4],[13.9,-2],[14.1,0],
                 [13.9,2],[13.6,4],[13,6],[12,8],[0,0] ]
        el_path=girar_path(el_path, giro)
        plt.plot(x,y, marker=el_path, markersize=18, color='None', linestyle='None',
            markeredgecolor='b', markeredgewidth= 1, alpha=0.8)


def pinta_malla(dominio):

    nelems= len(dominio.elems)
    if   nelems>220: npl=3
    elif nelems>70: npl=5
    elif nelems>20: npl=9
    else: npl=17

    if nelems < 80:
        # dibujo los nodos:
        for i,n in dominio.nodos.items():
            x,y = n.x, n.y
            plt.plot(x, y, 'd', color='grey', markersize=3)
            a = (np.random.random()-0.5) * dominio.anchox/22
            b = (np.random.random()-0.5) * dominio.anchoy/22
            plt.text(x+a, y+b, str(i), fontsize=9, ha='center', va='center')

        # dibujo los bordes en detalle y numero los elem:
        for i,e in dominio.elems.items():
            e.pinta_elem(colorin='black', ancholin=0.8, tipolin='', npl=npl)
            x,y= (e.xi[1]+e.xi[5])/2 , (e.yi[1]+e.yi[5])/2
            plt.text(x,y,str(i),fontsize=10, bbox=dict(facecolor='g', 
                    edgecolor='white', alpha=0.2, boxstyle='round'))
    else:
        if nelems < 510:
            # dibujo los nodos a pelo:
            for i,n in dominio.nodos.items():
                x,y = n.x, n.y
                plt.plot(x, y, 'd', color='grey', markersize=3)

        # dibujo los bordes a pelo:
        for i,e in dominio.elems.items():
            e.pinta_elem(colorin='black', ancholin=0.8, tipolin='', npl=npl)
    


def pinta_vistazo():
    global casos
    texto=''
    info_strv.set(texto)
    if not len(casos): 
        leer_gui()
        texto += 'A mesh has been created based on the given data.'
        ncasos_strv.set('1')

    if casos[0] == 1 or len(casos[0].dominio.nodos)==0 or len(casos[0].dominio.elems)==0:
        info_strv.set('BAD: not enough data to build a quick view.')
        casos=[]
        ncasos_strv.set('0')
        return(1)
    
    pinta_caso(0)
    
    texto += 'Quick view has been generated.'
    info_strv.set(texto)
    

def pinta_caso(icaso): 
    # Pinta un caso con sus cc. Necesita que venga ya con n_ext etc
    global casos
    caso = casos[icaso]

    plt.close(fig='Genesis data')
    plt.figure('Genesis data')
    free='free'

    pinta_malla (caso.dominio)
    
    # dibujo sustentacion -> nodos con u impuesto
    for i,lista in caso.ccpunto.u_imp.items():
        pseudo_n_ext=caso.dominio.nodos[i].pseudo_n_ext
        x,y= caso.dominio.nodos[i].x , caso.dominio.nodos[i].y
        vec,u1,u2 = lista
        if vec=='xy' or vec=='':  # nt no se permite en nodos sueltos
            if u1==free:
                v=np.array([0,1])
                proy = np.sign(np.dot(v, pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'apoyom', v*proy)
            elif u2==free:
                v=np.array([1,0])
                proy= np.sign(np.dot(v, pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'apoyom', v*proy)
            else:
                pinta_figurica(x,y, 'apoyof', [1,0])
        elif vec=='nt':
            texto = '"nt" is only valid for Zone boundary conditions'
            texto+= '(n_ext may be undefined at a single node)\n'
            texto+='You may manually provide a pseudo normal.'
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        else:
            vec=np.radians(float(vec)) 
            n=np.array([np.cos(vec), np.sin(vec)])
            if u1==free:
                t=np.array([n[1],-n[0]])
                proy= np.sign(np.dot(t,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'apoyom', t*proy )
            elif u2==free:
                proy= np.sign(np.dot(n,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'apoyom', n*proy)
            else:
                pinta_figurica(x,y, 'apoyof', [1,0])            
            
            
    # dibujo sustentacion -> resortes puntuales
    for i,lista in caso.ccpunto.kas.items():
        pseudo_n_ext=caso.dominio.nodos[i].pseudo_n_ext
        x,y= caso.dominio.nodos[i].x , caso.dominio.nodos[i].y
        vec,k1,k2 = lista
        if not(k1 or k2):
            texto = 'Error in punctual spring. Node '+str(i)
            texto+= ' has no associated stiffness.'
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        if vec=='xy' or vec=='': 
            if k1:          # en kas no hay free, es kx=0. & asinn
                v=np.array([1,0])
                proy= np.sign(np.dot(v,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'muelle', v*proy)
            if k2:
                v=np.array([0,1])
                proy=np.sign(np.dot(v,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'muelle', v*proy)
        elif vec=='nt':
            texto = '"nt" is only valid for Zone boundary conditions'
            texto+= '(n_ext may be undefined at a single node)\n'
            texto+='You may manually provide a pseudo normal.'
            messagebox.showinfo(message=texto,title='BAD',parent=v0)
            return(1)
        else:
            vec=np.radians(float(vec)) 
            n=np.array([np.cos(vec), np.sin(vec)])
            if k1:
                proy= np.sign(np.dot(n,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'muelle', n*proy)
            if k2:
                t=np.array([n[1],-n[0]])
                proy=np.sign(np.dot(t,pseudo_n_ext)) or 1
                pinta_figurica(x,y, 'muelle', t*proy)

    
    # asigno escala para dibujar luego xraya
    max_xraya = 0.
    for el in caso.dominio.elems.values():
        for ccz in el.cczona:
            if ccz[-2]=='xraya':
                for i in range(4,10):
                    if abs(ccz[i]) > max_xraya: max_xraya = abs(ccz[i])
    scal_xraya = (caso.dominio.anchox+caso.dominio.anchoy)/max_xraya/9 if max_xraya else 1.


    # dibujo cc de zona
    eps=(caso.dominio.anchox + caso.dominio.anchoy)/270  # un eps para dibujar
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]=='zu_imp': # es tipo zu_imp
                nodos_z= ccz[1:4]
                xi= [caso.dominio.nodos[i].x  for i in nodos_z]
                yi= [caso.dominio.nodos[i].y  for i in nodos_z]
                zona=Zona(xi,yi)
                zona.pinta_zona(desplacin=-eps)
                n,t = [],[]  # las n_ext en los tres nodos
                for psi in (0., 0.5, 1.): 
                    a,b=zona.n_ext(psi)
                    n.append(a)
                    t.append(b)
                if ccz[0]=='nt': 
                    if ccz[4]==free:
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'apoyom', t[i])
                    elif ccz[7]==free:
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'apoyom', n[i])
                    else:  # zona empotrada
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'apoyof', [1,0])
                elif ccz[0]=='xy' or ccz[0]=='' :
                    if ccz[4]==free:
                        for i in range(3):
                            a=[0,1] if n[i][1]>0. else [0,-1]
                            pinta_figurica(xi[i],yi[i], 'apoyom', a)
                    elif ccz[7]==free:
                        for i in range(3):
                            a=[1,0] if n[i][0]>0. else [-1,0]
                            pinta_figurica(xi[i],yi[i], 'apoyom', a)
                    else:  # zona empotrada
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'apoyof', [1,0])
                else:
                    g=np.radians(float(ccz[0]))
                    vecn=np.array([np.cos(g),  np.sin(g)])
                    vect=np.array([-vecn[1] , vecn[0]])
                    if ccz[4]==free:
                        for i in range(3):
                            proy= np.sign(np.dot(vect,n[i]))
                            pinta_figurica(xi[i],yi[i], 'apoyom', proy*vect)
                    elif ccz[7]==free:
                        for i in range(3):
                            proy= np.sign(np.dot(vecn,n[i]))
                            pinta_figurica(xi[i],yi[i], 'apoyom', proy*vecn)
                    else:  # zona empotrada
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'apoyof', [1,0])

            if ccz[-2]=='zkas':  # es de tipo zkas
                nodos_z= ccz[1:4]
                xi= [caso.dominio.nodos[i].x  for i in nodos_z]
                yi= [caso.dominio.nodos[i].y  for i in nodos_z]
                zona=Zona(xi,yi)
                zona.pinta_zona(desplacin=-eps)
                n,t = [],[]  # las n_ext en los tres nodos
                for psi in (0., 0.5, 1.): 
                    a,b=zona.n_ext(psi)
                    n.append(a)
                    t.append(b)
                if ccz[0]=='nt': 
                    if any (a != 0 for a in ccz[4:7]):
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'muelle', n[i])
                    if any (a != 0 for a in ccz[7:10]):
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'muelle', t[i])

                elif ccz[0]=='xy' or ccz[0]=='' :
                    if any (a != 0 for a in ccz[4:7]):
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'muelle', [np.sign(n[i][0]),0])
                    if any (a != 0 for a in ccz[7:10]):
                        for i in range(3):
                            pinta_figurica(xi[i],yi[i], 'muelle', [0,np.sign(n[i][1])])
                else:
                    g=np.radians(float(ccz[0]))
                    vecn=np.array([np.cos(g), np.sin(g)])
                    vect=np.array([-vecn[1] , vecn[0]])
                    if any (a != 0 for a in ccz[4:7]):
                        for i in range(3):
                            proy=np.sign(np.dot(n[i],vecn))
                            pinta_figurica(xi[i],yi[i], 'muelle', proy*vecn)
                    if any (a != 0 for a in ccz[7:10]):
                        for i in range(3):
                            proy=np.sign(np.dot(n[i],vect))
                            pinta_figurica(xi[i],yi[i], 'muelle', proy*vect)

            if ccz[-2]=='xraya':  # es de tipo xraya
                nodos_z= ccz[1:4]
                xi= [caso.dominio.nodos[i].x  for i in nodos_z]
                yi= [caso.dominio.nodos[i].y  for i in nodos_z]
                zona=Zona(xi,yi)
                p1i, p2i = ccz[4:7], ccz[7:10]
                n,t = [],[]  # sucesivamente, las n_ext & t en los 18 ptos
                x_dibu, y_dibu = [], []
                
                if any (i for i in p1i):
                    j=0
                    for psi in np.linspace(0.,1.,17):
                        a,b= zona.dimexy(psi)
                        n,t= zona.n_ext(psi)
                        valor=np.matmul(zona.Hzi(psi), p1i)
                        c= n*abs(valor)*scal_xraya
                        x_dibu.append(a+c[0])
                        y_dibu.append(b+c[1])

                        if j in [0,16]:
                            plt.plot([a, a+c[0]],[b, b+c[1]],'-',
                                    linewidth=0.8, color='r')
                        if j in [4,8,12]:
                            if ccz[0]=='nt':
                                if valor>0 :
                                    desde= [a,b]
                                    dxy=c
                                else:
                                    desde=[a+c[0], b+c[1]]
                                    dxy = -c 
                                plt.arrow(desde[0], desde[1], dxy[0], dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='r', length_includes_head=True)
                            elif ccz[0]=='xy':
                                kk=valor*scal_xraya
                                dxy=[kk, 0]
                                if (n[0]>0 and valor>0) or (n[0]<0 and valor<0):
                                    desde=[a,b]
                                else:
                                    desde=[a-kk, b]
                                plt.arrow(desde[0], desde[1], dxy[0] , dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='r', length_includes_head=True)
                            else:  # ccz[0] es un giro dado
                                giro=np.radians(float(ccz[0]))
                                vec=np.array([np.cos(giro), np.sin(giro)])
                                kk=valor*scal_xraya
                                dxy=kk*vec
                                if (np.dot(n,vec)>0 and valor>0) or (np.dot(n,vec)<0 and valor<0):
                                    desde=[a,b]
                                else:
                                    desde=[a-dxy[0], b-dxy[1]]
                                plt.arrow(desde[0], desde[1], dxy[0] , dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='r', length_includes_head=True)
                        j+=1
                    plt.plot(x_dibu,y_dibu,'-',linewidth=0.8, color='r')
            

                x_dibu, y_dibu = [], [] # sin herencias indebidas
                if any (i for i in p2i):
                    j=0
                    for psi in np.linspace(0.,1.,17):
                        a,b= zona.dimexy(psi)
                        n,t= zona.n_ext(psi)
                        valor=np.matmul(zona.Hzi(psi), p2i)
                        c= n*abs(valor)*scal_xraya
                        x_dibu.append(a+c[0])
                        y_dibu.append(b+c[1])

                        if j in [0,16]:
                            plt.plot([a, a+c[0]],[b, b+c[1]],'-',
                                    linewidth=0.8, color='orange', alpha=0.8)
                        if j in [4,8,12]:
                            if ccz[0]=='nt':
                                desde=[a,b]+n*abs(valor)*scal_xraya/2
                                dxy=t*valor*scal_xraya
                                plt.arrow(desde[0], desde[1], dxy[0], dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='orange', length_includes_head=True)
                            elif ccz[0]=='xy':
                                kk=valor*scal_xraya
                                dxy=[0, kk]
                                if (n[1]>0 and valor>0) or (n[1]<0 and valor<0):
                                    desde=[a,b]
                                else:
                                    desde=[a, b-kk]
                                plt.arrow(desde[0], desde[1], dxy[0] , dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='orange', length_includes_head=True)
                            else:  # ccz[0] es un giro dado
                                giro=np.radians(float(ccz[0]))
                                vec=np.array([-np.sin(giro), np.cos(giro)])
                                kk=valor*scal_xraya
                                dxy=kk*vec
                                if (np.dot(n,vec)>0 and valor>0) or (np.dot(n,vec)<0 and valor<0):
                                    desde=[a,b]
                                else:
                                    desde=[a-dxy[0], b-dxy[1]]
                                plt.arrow(desde[0], desde[1], dxy[0] , dxy[1],
                                        head_width=2*eps, linestyle='solid', linewidth=1, 
                                        overhang=0.6, color='orange', length_includes_head=True)
                        j+=1
                    plt.plot(x_dibu,y_dibu,'-',linewidth=0.8, color='orange', alpha=0.8)
            


    # dibujo cargas -> fuerzas
    
    max_F = 0.
    for lista in caso.ccpunto.fuerzas.values():
        if abs(lista[1])> max_F : max_F=abs(lista[1])
        if abs(lista[2])> max_F : max_F=abs(lista[2])
    scal_F = 0.1*(caso.dominio.anchox+caso.dominio.anchoy)/max_F if max_F else 1.

    for i,lista in caso.ccpunto.fuerzas.items():
        pseudo_n_ext= caso.dominio.nodos[i].pseudo_n_ext
        x=caso.dominio.nodos[i].x
        y=caso.dominio.nodos[i].y
        if lista[1]:
            if lista[0]=='xy':
                dxy=lista[1]*scal_F
                proy=np.sign(np.dot([dxy,0],pseudo_n_ext)) or 1
                desde =[x,y] if proy>0 else [x-dxy, y] 
                plt.arrow(desde[0], desde[1], dxy , 0,
                        head_width=3*eps, linestyle='solid', linewidth=1.6, 
                        overhang=0.6, color='m', length_includes_head=True)
            elif lista[0]=='nt':
                print('"nt" is not valid for puntual loads')
                print('(the normal may be undefined at a single point).')
                return(1)
            else:
                giro=np.radians(lista[0])
                vec=np.array([np.cos(giro), np.sin(giro)])
                dxy=lista[1]*scal_F*vec
                proy=np.sign( np.dot(dxy,pseudo_n_ext) ) or 1
                desde= [x,y] if proy>0 else [x-dxy[0], y-dxy[1]] 
                plt.arrow(desde[0], desde[1], dxy[0], dxy[1],
                        head_width=3*eps, linestyle='solid', linewidth=1.6, 
                        overhang=0.6, color='m', length_includes_head=True)

        if lista[2]:
            if lista[0]=='xy':
                dxy=lista[2]*scal_F
                proy=np.sign(np.dot([0,dxy],pseudo_n_ext)) or 1
                desde=[x,y] if proy>0 else [x, y-dxy]
                plt.arrow(desde[0], desde[1], 0, dxy,
                        head_width=3*eps, linestyle='solid', linewidth=1.6, 
                        overhang=0.6, color='m', length_includes_head=True)
            elif lista[0]=='nt':
                print('"nt" is not valid for puntual loads')
                print('(the normal may be undefined at a single point).')
                return(1)
            else:
                giro=np.radians(lista[0])
                vec=np.array([-np.sin(giro), np.cos(giro)])
                dxy=lista[2]*scal_F*vec
                proy=np.sign( np.dot(dxy,pseudo_n_ext) ) or 1
                desde =[x,y] if proy>0 else [x-dxy[0], y-dxy[1]] 
                plt.arrow(desde[0], desde[1], dxy[0], dxy[1],
                        head_width=3*eps, linestyle='solid', linewidth=1.6, 
                        overhang=0.6, color='m', length_includes_head=True)    
    
    plt.show(block=False)




def refinar4():
    global casos  
    
    try:
        caso=casos[-1]  # se requiere un caso con al menos los vecinos() puestos
    except:
        info_strv.set('BAD: No base mesh to refine.')
        return(1)
    
    if caso==1 or len(caso.dominio.nodos)==0 or len(caso.dominio.elems)==0:
        info_strv.set('BAD: No base mesh data to refine.')
        return(1)        

    nodo_max=0
    for i in caso.dominio.nodos.keys():
        if i > nodo_max: nodo_max=i
    nodo_max +=1

    # iniciar nodos_new con clones de los nodos del caso anterior, pero objetos nuevos!
    nodos_new = {}
    for i,n in caso.dominio.nodos.items():
        nodos_new[i]=deepcopy(n)
        
    
    r3= np.sqrt(3.)
    pull=[ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], 
           [0.25, 0.25*r3], [0.75, 0.75*r3],  [0.5, r3], [-0.5, r3], 
           [-0.75, 0.75*r3], [-0.25,0.25*r3], [0,r3/2], [0.25, 0.75*r3], 
           [-0.25, 0.75*r3]    ]
    # son los psi,eta de los nodos nuevos (intermedios) a crear. 
    # Hay =0 en los antiguos: que existan para facilitar indexacion.

    # iniciar el huevo con los datos del propio elem
            # huevo[0] tiene los nodos que derivaran del elem
            # huevo[1], huevo[2] tienen las x,y de esos nodos
    for el in caso.dominio.elems.values():
        el.huevo= [ [],[],[] ]
        for i in el.nodse: el.huevo[0].append(i)
        for x in el.xi: el.huevo[1].append(x)
        for y in el.yi: el.huevo[2].append(y)
        for j in range(9): 
            el.huevo[0].append(-1) # relevante
            el.huevo[1].append(-1)
            el.huevo[2].append(-1) # para que existan

    for iel,el in caso.dominio.elems.items():
        # rellenar lo que no este relleno del huevo:
        for i in range(6,15):
            if  el.huevo[0][i] == -1 :
                el.huevo[0][i] = nodo_max
                x,y = el.dimexy( pull[i][0], pull[i][1] )
                el.huevo[1][i] = x
                el.huevo[2][i] = y
                nodos_new[nodo_max]=Nodo(x,y)
                nodo_max += 1
        # rellenar en el vecino lo que tenga sin rellenar
        for ilado in range(3):
            ivecino = el.vecinos[ilado][0]
            if ivecino != -1 :
                vecino = caso.dominio.elems[ivecino]
                sulado = el.vecinos[ilado][1]
                su_n1= 6+sulado*2
                su_n2= su_n1+1
                mi_n1= 6+ilado*2
                mi_n2= mi_n1+1
                if vecino.huevo[0][su_n1] == -1:
                    vecino.huevo[0][su_n1]=el.huevo[0][mi_n2]
                    vecino.huevo[1][su_n1]=el.huevo[1][mi_n2]
                    vecino.huevo[2][su_n1]=el.huevo[2][mi_n2]
                    vecino.huevo[0][su_n2]=el.huevo[0][mi_n1]
                    vecino.huevo[1][su_n2]=el.huevo[1][mi_n1]
                    vecino.huevo[2][su_n2]=el.huevo[2][mi_n1]

    # explotar los huevos:
    
    elems_new = {}
    iel_new = 1 
    zonatonta=Zona([0,0,0],[0,0,0]) # es para saber zona.Hzi(0.25) & 0.75 
    H025, H075 = zonatonta.Hzi(0.25), zonatonta.Hzi(0.75)
    
    for iel,el in caso.dominio.elems.items():
        
        nodse_new=[ el.huevo[0][0], el.huevo[0][6], el.huevo[0][1], 
                    el.huevo[0][12], el.huevo[0][5], el.huevo[0][11] ]
        e=Elem(mat=el.mat, nodse=nodse_new)
        for ccz in el.cczona:
            ccz_new=[]
            if ccz[-1]==0:  # lado 0 de ambos
                try: val1=np.dot(H025, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H025, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[0], e.nodse[1], e.nodse[2],
                         ccz[4], val1, ccz[5], ccz[7], val2, ccz[8], ccz[-2], 0 ]
            elif ccz[-1]==2:  # lado 2 de ambos
                try: val1=np.dot(H075, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H075, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[4], e.nodse[5], e.nodse[0],
                         ccz[5], val1, ccz[6], ccz[8], val2, ccz[9], ccz[-2], 2 ]
            if len(ccz_new): e.cczona.append(ccz_new)
        elems_new[iel_new]=e
        iel_new +=1


        nodse_new=[ el.huevo[0][1], el.huevo[0][7], el.huevo[0][2], 
                    el.huevo[0][8], el.huevo[0][3], el.huevo[0][13] ]
        e=Elem(mat=el.mat, nodse=nodse_new)
        for ccz in el.cczona:
            ccz_new=[]
            if ccz[-1]==0:  # lado 0 de ambos
                try: val1=np.dot(H075, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H075, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[0], e.nodse[1], e.nodse[2],
                         ccz[5], val1, ccz[6], ccz[8], val2, ccz[9], ccz[-2], 0 ]
            elif ccz[-1]==1:  # lado 1 de ambos
                try: val1=np.dot(H025, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H025, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[2], e.nodse[3], e.nodse[4],
                         ccz[4], val1, ccz[5], ccz[7], val2, ccz[8], ccz[-2], 1 ]
            if len(ccz_new): e.cczona.append(ccz_new)
        elems_new[iel_new]=e
        iel_new +=1


        nodse_new=[ el.huevo[0][5], el.huevo[0][12], el.huevo[0][1], 
                    el.huevo[0][13], el.huevo[0][3], el.huevo[0][14] ]
        e=Elem(nodse=nodse_new)
        elems_new[iel_new]=e  # es el interior, no hay que comprobar cczona
        iel_new +=1

 
        nodse_new=[ el.huevo[0][5], el.huevo[0][14], el.huevo[0][3], 
                    el.huevo[0][9], el.huevo[0][4], el.huevo[0][10] ]
        e=Elem(mat=el.mat, nodse=nodse_new)
        for ccz in el.cczona:
            ccz_new=[]
            if ccz[-1]==1:  # lado 1 de ambos
                try: val1=np.dot(H075, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H075, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[2], e.nodse[3], e.nodse[4],
                         ccz[5], val1, ccz[6], ccz[8], val2, ccz[9], ccz[-2], 1 ]
            elif ccz[-1]==2:  # lado 2 de ambos
                try: val1=np.dot(H025, ccz[4:7])
                except TypeError: val1='free'
                try: val2=np.dot(H025, ccz[7:10])
                except TypeError: val2='free'
                ccz_new=[ccz[0], e.nodse[4], e.nodse[5], e.nodse[0],
                         ccz[4], val1, ccz[5], ccz[7], val2, ccz[8], ccz[-2], 2 ]
            if len(ccz_new): e.cczona.append(ccz_new)
        elems_new[iel_new]=e
        iel_new +=1
    
    dominio_new = Dominio(nodos_new, elems_new, caso.dominio.ctes, caso.dominio.tplana)
    dominio_new.pon_basicos()
    dominio_new.pon_extremos()
    dominio_new.pon_vecinos() # solo este (& basicos) es imprescindible para refinar
    dominio_new.pon_contornos()
    dominio_new.pon_pseudo_n_ext()
 
    casos.append(Caso(dominio_new, caso.ccpunto))

    nn, ne = len(dominio_new.nodos), len(dominio_new.elems)
    texto= f'Mesh {len(casos)-1} generated. No errors: {nn} nodes & {ne} elements'
    ncasos_strv.set(str(len(casos)))
    info_strv.set(texto)
    print('domain has {:} nodes & {:} elems'.format(nn, ne))
    plt.close(fig='Refined mesh')
    
    confirmacion = True
    if ne > 520:
        texto =f'Mesh with {nn} nodes & {ne} elements correctly generated.'
        texto+=' Wanna plot that?'
        confirmacion= messagebox.askyesno(message= texto, icon='question',
                    detail='        ("please be patient")', title='Draw', default='no')
    if confirmacion:
        plt.figure('Refined mesh')
        pinta_malla(casos[-1].dominio)
        plt.show(block=False)




def motor_calculo(i_di):
    global casos, info_strv
    caso=casos[i_di]
    
    
    #salida_prolija(v0, i_di) # solo en pruebas
    
    print('\nCore Calculation has been invoked with:')
    print(f'Mesh number = {i_di}')
    nn,ne= len(caso.dominio.nodos), len(caso.dominio.elems)
    print(f'Number of nodes & elements = {nn}  {ne}')
    
    progreso='Building matrices...'
    info_strv.set(progreso)
    print(progreso)
    
    # ponemos las matrices D para cada material
    matricesD={}
    for imat,mat in caso.dominio.ctes.items():
        E, nu= mat[0], mat[1]
        if caso.dominio.tplana: # sera =1, TP
            a=E/(1-nu*nu)
            D = np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
            D *= a
        else: # sera =0, DP
            a, b = E/((1.+nu)*(1.-2*nu)), (1.-nu)
            D = np.array([[b,nu,0],[nu,b,0],[0,0,(1-2*nu)/2]])
            D *= a
        matricesD[imat]=D


    # ponemos la inclinacion de apoyo en los nodos[].apoyo_incl que la tengan
    for el in caso.dominio.elems.values():
        for ccu in el.cczona:
            if ccu[-2]== 'zu_imp':
                if ccu[0]=='nt':
                    xi=[caso.dominio.nodos[ccu[1]].x, caso.dominio.nodos[ccu[2]].x, 
                        caso.dominio.nodos[ccu[3]].x ]
                    yi=[caso.dominio.nodos[ccu[1]].y, caso.dominio.nodos[ccu[2]].y, 
                        caso.dominio.nodos[ccu[3]].y ]
                    zona=Zona(xi,yi)
                    n,t = zona.n_ext(0.)
                    alfa=np.arctan2(n[1],n[0])
                    caso.dominio.nodos[ccu[1]].apoyo_incl=alfa
                    n,t = zona.n_ext(0.5)
                    alfa=np.arctan2(n[1],n[0])
                    caso.dominio.nodos[ccu[2]].apoyo_incl=alfa
                    n,t = zona.n_ext(1.)
                    alfa=np.arctan2(n[1],n[0])
                    caso.dominio.nodos[ccu[3]].apoyo_incl=alfa
                elif ccu[0]=='xy':
                    pass
                else:  # es angulo dado
                    alfa= np.radians(float(ccu[0]))
                    caso.dominio.nodos[ccu[1]].apoyo_incl=alfa
                    caso.dominio.nodos[ccu[2]].apoyo_incl=alfa
                    caso.dominio.nodos[ccu[3]].apoyo_incl=alfa
    
    for inodo, ccu in caso.ccpunto.u_imp.items(): # prevalecera lo del puntual
        try:
            alfa=np.radians(float(ccu[0]))
            caso.dominio.nodos[inodo].apoyo_incl= alfa
        except ValueError:
            pass    
    
    # asignamos gdl (fila & column en K) a los nodos:
    gdl_nodos, gdl_max = {}, 0
    for inodo in caso.dominio.nodos.keys():
        gdl_nodos[inodo]= gdl_max
        gdl_max += 2
    
    
    # montamos la matriz de rigidez en formato disperso coo
    # y los terminos de carga de volumen y de temperatura:

    nnodos = len(caso.dominio.nodos)
    ldatos, li, lj = [],[],[]
    f_cargas=np.zeros(2*nnodos)
    
    for iel, el in caso.dominio.elems.items():
        matrizD=matricesD[el.mat]
        f_aportV, f_aportT = np.zeros((6,2)), np.zeros((6,2)) 
        Kij_el = np.zeros(144).reshape(6,6,2,2)
        
        for igaus in range(6):
            jaco, Ni_psi, Ni_eta = el.dimeJacoss(el.x_Gaus[igaus], el.y_Gaus[igaus])
            jacoinv= np.linalg.inv(jaco)
            psi_x, psi_y, eta_x, eta_y = jacoinv.reshape(4)
            Ni_x = Ni_psi*psi_x + Ni_eta*eta_x
            Ni_y = Ni_psi*psi_y + Ni_eta*eta_y # de cada una de las 6 ff
            
            for inode in range(6):
                LNi=np.array([ [Ni_x[inode],0],[0, Ni_y[inode]],[Ni_y[inode], Ni_x[inode]] ])
                for jnode in range(6):
                    LNj=np.array([ [Ni_x[jnode],0],[0, Ni_y[jnode]],[Ni_y[jnode], Ni_x[jnode]] ])
                    Kij_el[inode,jnode] += np.matmul(
                        np.matmul(LNi.T,matrizD), LNj) * el.w_Gaus[igaus]* np.linalg.det(jaco)
                
                if len(caso.ccpunto.xgrande):
                    x,y= el.dimexy(el.x_Gaus[igaus], el.y_Gaus[igaus])
                    fVol=np.array([eval(caso.ccpunto.xgrande[0]), 
                                   eval(caso.ccpunto.xgrande[1])])
                    Ni= el.dimeHi(el.x_Gaus[igaus], el.y_Gaus[igaus])
                    f_aportV[inode] += Ni[inode]* fVol* el.w_Gaus[igaus]* np.linalg.det(jaco)

                if len(caso.ccpunto.temperatura): 
                    x,y= el.dimexy(el.x_Gaus[igaus], el.y_Gaus[igaus])
                    Tem= eval(caso.ccpunto.temperatura[0]) 
                    a= caso.dominio.ctes[el.mat][2] * Tem
                    if not caso.dominio.tplana :
                        a *= (1+caso.dominio.ctes[el.mat][1])
                    eps0= np.array([a,a,0])
                    f_aportT[inode] +=  (np.matmul( np.matmul(LNi.T,matrizD), eps0)* 
                                         el.w_Gaus[igaus]* np.linalg.det(jaco))
        
        
        # Kij_el (del elem) montada en ejes xy. La coloco en K en formato coo
        # f_aportT, f_aportV del elem calculadas. Las llevo a f:
        
        for inode in range(6):
            nodoi= el.nodse[inode]
            gdli= gdl_nodos[nodoi]
            for jnode in range(6):
                nodoj = el.nodse[jnode]
                gdlj = gdl_nodos[nodoj]
                K2x2= Kij_el[inode, jnode]

                # le cambio de ejes aqui: es mas facil aunque sea menos optimo
                if caso.dominio.nodos[nodoi].apoyo_incl:
                    alfa= caso.dominio.nodos[nodoi].apoyo_incl
                    s,c= np.sin(alfa), np.cos(alfa)
                    a=np.array([[c,s],[-s,c]])
                    K2x2= np.matmul(a,K2x2)
                if caso.dominio.nodos[nodoj].apoyo_incl:
                    alfa= caso.dominio.nodos[nodoj].apoyo_incl
                    s,c= np.sin(alfa), np.cos(alfa)
                    a=np.array([[c,-s],[s,c]])
                    K2x2= np.matmul(K2x2,a)

                ldatos.append(K2x2[0][0])
                li.append(gdli)
                lj.append(gdlj)

                ldatos.append(K2x2[0][1])
                li.append(gdli)
                lj.append(gdlj+1)

                ldatos.append(K2x2[1][0])
                li.append(gdli+1)
                lj.append(gdlj)

                ldatos.append(K2x2[1][1])
                li.append(gdli+1)
                lj.append(gdlj+1)
            
            if len(caso.ccpunto.xgrande):
                f_cargas[gdli]   += f_aportV[inode][0] 
                f_cargas[gdli+1] += f_aportV[inode][1] 
            if len(caso.ccpunto.temperatura):
                f_cargas[gdli]   += f_aportT[inode][0]
                f_cargas[gdli+1] += f_aportT[inode][1]



    # Aportaciones a K de los resortes (en xy + con giro de apoyo si toca)
    for nodoi, lista in caso.ccpunto.kas.items():
        gdli= gdl_nodos[nodoi]
        tensork=np.array([ [lista[1],0], [0,lista[2]] ])
        try:  # cambiarlo a xy si se dio en giradas
            alfa= np.radians(lista[0])
            s,c = np.sin(alfa), np.cos(alfa)
            a= np.array([ [c,-s],[s,c] ])
            tensork = np.matmul( np.matmul(a,tensork), a.T )
        except (TypeError, ValueError):
            pass
        # tensork esta en xy. Si es en apoyo inclinado girar again
        if caso.dominio.nodos[nodoi].apoyo_incl:
            alfa= caso.dominio.nodos[nodoi].apoyo_incl
            s,c= np.sin(alfa), np.cos(alfa)
            a=np.array([[c,s],[-s,c]])
            tensork= np.matmul( np.matmul(a,tensork), a.T)

        ldatos.append(tensork[0][0])
        li.append(gdli)
        lj.append(gdli)

        ldatos.append(tensork[0][1])
        li.append(gdli)
        lj.append(gdli+1)

        ldatos.append(tensork[1][0])
        li.append(gdli+1)
        lj.append(gdli)

        ldatos.append(tensork[1][1])
        li.append(gdli+1)
        lj.append(gdli+1)
    
    # aportaciones a K de las zonas de resortes
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]=='zkas':
                k1i, k2i = ccz[4:7], ccz[7:10]
                xi, yi=[], []
                tensores_i= np.zeros((3,2,2))
                for inodo in range(3):
                    xi.append(caso.dominio.nodos[ccz[inodo+1]].x)
                    yi.append(caso.dominio.nodos[ccz[inodo+1]].y)
                    tensores_i[inodo]= np.array([[k1i[inodo],0], [0,k2i[inodo]]])
                zona= Zona(xi, yi)

                # lo primero poner los tres tensores_i[] en xy si no lo estan
                # pq en una zona nt pueden estar cada uno en unos ejes
                if ccz[0]=='xy':
                    pass
                elif ccz[0]=='nt':
                    for inodo in range(3):
                        psi= inodo*0.5
                        n,t = zona.n_ext(psi)
                        s,c = n[1], n[0]
                        a= np.array([ [c,-s],[s,c] ])
                        tensores_i[inodo] = np.matmul( 
                                        np.matmul(a,tensores_i[inodo]), a.T )
                else:  # sera un angulo
                    alfa=np.radians(ccz[0])
                    s,c = np.sin(alfa), np.cos(alfa)
                    a= np.array([ [c,-s],[s,c] ])
                    for inodo in range(3):
                        tensores_i[inodo] = np.matmul( 
                                        np.matmul(a,tensores_i[inodo]), a.T )
                
                # ahora hacemos las integrales con las k's en xy:
                k_aport= np.zeros((3,3,2,2))
                for igaus in range(3):
                    psi= zona.x_Gaus[igaus]
                    b= np.sqrt( np.dot(zona.Hzi_psi(psi), xi)**2 + 
                                np.dot(zona.Hzi_psi(psi), yi)**2)
                    Ni_en_psi= zona.Hzi(psi)
                    k_en_psi= np.tensordot (Ni_en_psi, tensores_i, [0,0])
                    for inodo in range(3):
                        for jnodo in range(3):
                            k2x2 = Ni_en_psi[inodo]*Ni_en_psi[jnodo]*k_en_psi * b
                            k_aport[inodo,jnodo] += k2x2 * zona.w_Gaus[igaus]
                
                # ensamblo las 3x3 k_aport de 2x2, giradas si hay apoyo_incl
                for inodo in range(3):
                    nodoi = ccz[inodo+1]
                    gdli = gdl_nodos[nodoi]
                    for jnodo in range(3):
                        nodoj = ccz[jnodo+1]
                        gdlj = gdl_nodos[nodoj]
                        k2x2= k_aport[inodo][jnodo]
                        if caso.dominio.nodos[nodoi].apoyo_incl:
                            alfa= caso.dominio.nodos[nodoi].apoyo_incl
                            s,c = np.sin(alfa), np.cos(alfa)
                            a= np.array([ [c,s],[-s,c] ])
                            k2x2= np.matmul(a,k2x2)
                        if caso.dominio.nodos[nodoj].apoyo_incl:
                            alfa= caso.dominio.nodos[nodoj].apoyo_incl
                            s,c = np.sin(alfa), np.cos(alfa)
                            a= np.array([ [c,-s],[s,c] ])
                            k2x2= np.matmul(k2x2,a)

                        ldatos.append(k2x2[0][0])
                        li.append(gdli)
                        lj.append(gdlj)

                        ldatos.append(k2x2[0][1])
                        li.append(gdli)
                        lj.append(gdlj+1)

                        ldatos.append(k2x2[1][0])
                        li.append(gdli+1)
                        lj.append(gdlj)

                        ldatos.append(k2x2[1][1])
                        li.append(gdli+1)
                        lj.append(gdlj+1)

    
    # Matriz K_coo con elementos girados. La mantendre para uso posterior
    K_coo= coo_matrix((ldatos,(li,lj)))
    
    # Las matrices elementales se han guardado *giradas* en formato coo con 
    # repeticiones. Se cuenta con que se sumen automaticamente al pasar a csr.
    # Las aportaciones de xraya & T se han guardado *sin girar* en f.
    
    # Aportaciones de fuerzas: 
    for nodoi, lista  in caso.ccpunto.fuerzas.items():
        gdli= gdl_nodos[nodoi]
        f_aportF= np.array([ lista[1], lista[2] ])
        try:
            alfa=np.radians(lista[0])
        except (TypeError, ValueError):
            alfa=0
        if alfa: # transformamos a xy
            s,c= np.sin(alfa), np.cos(alfa)
            a= np.array([ [c,-s],[s,c] ])
            f_aportF = np.matmul(a, f_aportF)
        f_cargas[gdli]   += f_aportF[0]
        f_cargas[gdli+1] += f_aportF[1]
    
    # Aportaciones de xraya: 
    for iel, el in caso.dominio.elems.items():
        for ccz in el.cczona:
            if ccz[-2]=='xraya':
                lado= ccz[-1]
                ns= ccz[1:4]
                if lado<2:
                    xi = np.array(el.xi[lado*2:lado*2+3])
                    yi = np.array(el.yi[lado*2:lado*2+3])
                elif lado ==2:
                    xi = np.array([ el.xi[4], el.xi[5], el.xi[0] ])
                    yi = np.array([ el.yi[4], el.yi[5], el.yi[0] ])
                zona=Zona(xi, yi)
                xr1, xr2 = ccz[4:7], ccz[7:10]
                # ponemos en xy si hace falta:
                if ccz[0]=='xy':
                    pass
                elif ccz[0]=='nt':
                    for i in range(3):
                        psi= float(0.5*i)
                        n,t = zona.n_ext(psi)
                        a= np.array([ [n[0], -n[1]],[n[1], n[0]] ])
                        b= np.array([xr1[i], xr2[i]])
                        kk= np.matmul(a,b)
                        xr1[i], xr2[i] = kk[0], kk[1]
                else:  # sera angulo dado:
                    alfa = np.radians(ccz([0]))
                    s,c= np.sin(alfa), np.cos(alfa)
                    a= np.array([ [c,-s],[s,c] ])
                    for i in range(3):
                        b= np.array([xr1[i], xr2[i]])
                        kk= np.matmul(a,b)
                        xr1[i], xr2[i] = kk[0], kk[1]
                
                f_aportXr=np.zeros((3,2))
                for igaus in range(3):
                    psi= zona.x_Gaus[igaus]
                    a= np.sqrt( np.dot(zona.Hzi_psi(psi), xi)**2 + 
                                np.dot(zona.Hzi_psi(psi), yi)**2)
                    a *= zona.w_Gaus[igaus]
                    Ni= zona.Hzi(psi)
                    xr=np.array([ np.dot(Ni,xr1), np.dot(Ni, xr2) ])
                    for inodo in range(3):
                        N= Ni[inodo]
                        b= N*xr*a
                        f_aportXr[inodo] += b
                # calculada aportacion de zona. La ensamblo:
                for inodo in range(3):
                    nodoi= ns[inodo]
                    gdli= gdl_nodos[nodoi]
                    f_cargas[gdli]   += f_aportXr[inodo][0]
                    f_cargas[gdli+1] += f_aportXr[inodo][1]

    
    # Girar los terminos de f_cargas con apoyo_incl
    for i, nodo in caso.dominio.nodos.items():
        if nodo.apoyo_incl:
            alfa = nodo.apoyo_incl
            s,c= np.sin(alfa), np.cos(alfa)
            a= np.array([[c,s],[-s,c]])
            gdli = gdl_nodos[i]
            f2x1= np.array([ f_cargas[gdli], f_cargas[gdli+1] ])
            b=np.matmul(a, f2x1)
            f_cargas[gdli] = b[0]
            f_cargas[gdli+1] = b[1]
    
    
    a_despl, gdl_obviar= np.zeros(2*nnodos), []
    
    # Tratamiento de las cc en desplazamientos de zona. Lo hago antes que 
    # las ccpunto porque estas últimas deben prevalecer si hay conflicto.
    for el in caso.dominio.elems.values():
        for ccz in el.cczona:
            if ccz[-2]=='zu_imp':
                ni, u1i, u2i = ccz[1:4], ccz[4:7], ccz[7:10]
                for i in range(3):
                    gdli=gdl_nodos[ni[i]]
                    if u1i[i] != 'free':
                        a_despl[gdli]=u1i[i]
                        gdl_obviar.append(gdli)
                    if u2i[i] != 'free':
                        a_despl[gdli]=u2i[i]
                        gdl_obviar.append(gdli+1)
    
    
    # Tratamiento de las u_imp de punto.
    for inodo, lista in caso.ccpunto.u_imp.items():
        gdli= gdl_nodos[inodo]
        if lista[1] != 'free':
            a_despl[gdli]=lista[1]
            gdl_obviar.append(gdli)
        if lista[2] != 'free':
            a_despl[gdli+1]=lista[2]
            gdl_obviar.append(gdli+1)
    
    
    K_operar=K_coo.tocsr()
    
    # antes de desnudar K paso a f los desplaz conocidos:
    f_operar= f_cargas - K_operar*a_despl
    
    # desnudo K & f:
    gdl_keep= list(set(range(0,2*nnodos)) - set(gdl_obviar))
    K_operar = K_operar[gdl_keep, :]
    K_operar = K_operar[:, gdl_keep]
    f_operar = f_operar[gdl_keep]

    progreso += 'solving...'
    info_strv.set(progreso)
    print('solving...')
    
    # resuelvo:
    a_soluc= spsolve(K_operar, f_operar)
    
    progreso +='stress calculation...'
    info_strv.set(progreso)
    print('stress calculation...')

    # reconstruyo a_despl con los resultados:
    i=0
    for igdl in range (2*nnodos):
        if igdl not in gdl_obviar:
            a_despl[igdl] = a_soluc[i]
            i +=1
    # y lo guardo en los nodos, pero en xy
    for nodoi, nodo in caso.dominio.nodos.items():
        u= np.array([ a_despl[gdl_nodos[nodoi]], a_despl[gdl_nodos[nodoi]+1] ])
        if nodo.apoyo_incl:
            alfa= nodo.apoyo_incl # ya esta en radianes
            c,s= np.cos(alfa), np.sin(alfa)
            a= np.array([[c,-s],[s,c]])
            u= np.matmul(a, u)
        nodo.ux, nodo.uy = u[0], u[1]
    
    # calculo las incognitas de f & compruebo las dadas:
    f_soluc = K_coo * a_despl
    for igdl in range (2*nnodos):
        if igdl in gdl_obviar:
            f_cargas[igdl]=f_soluc[igdl]
        else:
            err = f_cargas[igdl] - f_soluc[igdl]
            if abs(err)> 1.e-4:         # es una medida absoluta, *** chapucilla...
                print(f'\n bad re-sustitution in f[{igdl}] :')
                print('given= {:8.2e}  ,  calculated= {:8.2e}'.format(f_cargas[igdl], f_soluc[igdl]))


    # calculo de tensiones: elem.sigma(6,4): comprobe en la v2 que las tensiones
    # del elemento estaran en un plano de forma natural, tanto en los pGaus como
    # donde quiera calcularlas. Con el lio de calcular en pGaus, ajustar plano &
    # extrapolar obtengo lo mismo que si evaluo en los nodos directamente. 
    # Asi que evaluo en los nodos.
    # La razon es clara: si u es parabolico sigma sera lineal (en el elem)
    
    for n in caso.dominio.nodos.values(): n.sigma= np.zeros(4) 
    for el in caso.dominio.elems.values(): el.sigma= np.zeros((6,4)) # por si es revisita
    
    for iel, el in caso.dominio.elems.items():
        matrizD=matricesD[el.mat]
        for inodo in range (6):
            x_, y_ = el.x_[inodo], el.y_[inodo]
            jaco, Ni_psi, Ni_eta = el.dimeJacoss(x_, y_)
            jacoinv= np.linalg.inv(jaco)
            psi_x, psi_y, eta_x, eta_y = jacoinv.reshape(4)
            Ni_x = Ni_psi*psi_x + Ni_eta*eta_x
            Ni_y = Ni_psi*psi_y + Ni_eta*eta_y # de cada una de las 6 ff
            LNa = np.zeros(3)
            for jnodo in range(6):
                nodoj= el.nodse[jnodo]
                a= np.array([ caso.dominio.nodos[nodoj].ux, 
                              caso.dominio.nodos[nodoj].uy ])
                LN= np.array([ [Ni_x[jnodo],      0], 
                               [0,       Ni_y[jnodo]],
                               [Ni_y[jnodo], Ni_x[jnodo]] ])
                LNa += np.matmul(LN, a)
                
            if len(caso.ccpunto.temperatura): 
                x,y = el.dimexy(x_, y_)
                Tem= eval(caso.ccpunto.temperatura[0]) 
                b= caso.dominio.ctes[el.mat][2] * Tem
                if not caso.dominio.tplana :
                    b *= (1+caso.dominio.ctes[el.mat][1])
                eps0= np.array([b,b,0])
                LNa -= eps0
            s = np.matmul(matrizD, LNa) # las tres tensiones en el plano
            
            # la de von mises:
            s33=0 if caso.dominio.tplana else caso.dominio.ctes[el.mat][1]*(s[0]+s[1])
            svm = (s[0]-s[1])**2+ (s[0]-s33)**2+ (s[1]-s33)**2 + 6*s[2]**2
            svm = np.sqrt(svm/2)
            
            el.sigma[inodo][0:3] = s
            el.sigma[inodo][3] = svm

        # Las tensiones nodales del elem estan. Toca guardarlo + en los nodos:
        for inodo in range(6):
            nodoi= el.nodse[inodo]
            caso.dominio.nodos[nodoi].sigma += el.sigma[inodo]

    # tensiones promediadas de nodo xx, yy, xy, vm, & tensiones principales:
    for ni, n in caso.dominio.nodos.items():
        n.sigma /= len(n.elemn)  # para promediar

        tensor= np.array( [  [n.sigma[0], n.sigma[2]] , [n.sigma[2], n.sigma[1]]  ])
        try:
            valores, vectores = np.linalg.eigh(tensor)
            vectores = vectores.T  # pq lo da por columnas
        except LinAlgError:
            print('Unable to calculate stress eigen-things for node', ni)
        n.s_prles=[ valores[1], vectores[1], valores[0], vectores[0]  ]
                        # lo da en ascendente, por eso invierto el orden

    caso.calculado=True  # marca el caso para no volverlo a calcular



    # Salida de resultados provisional en texto. Muestra solo los 
    # nodos del caso[0], aunque los resultados son del caso resuelto:
    
    print('\n'+'#'*50)
    print('Results ')
    n= len(casos[0].dominio.nodos)
    i=0
    print('\nNodal displacements (in xy, whether a tilted support is present or not)')
    for inodo, nodo in caso.dominio.nodos.items():
        print('{:3d}  {:12.4e}  {:12.4e}'.format(inodo, nodo.ux, nodo.uy))
        i += 1
        if i==n: break
    i=0
    print('\nTerms of f (reactions from support items except from springs):')
    for inodo, nodo in caso.dominio.nodos.items():
        igdl = gdl_nodos[inodo]
        print('{:3d}  {:12.4e}  {:12.4e}'.format(inodo, 
                                        f_cargas[igdl], f_cargas[igdl+1]))
        i += 1
        if i==n: break

    print('\n Stress at nodes (from average):')
    print('node   sxx        syy        sxy       svm')
    for nodoi in casos[0].dominio.nodos.keys():
        print('{:3d} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(
            nodoi,  caso.dominio.nodos[nodoi].sigma[0],
                    caso.dominio.nodos[nodoi].sigma[1],
                    caso.dominio.nodos[nodoi].sigma[2],
                    caso.dominio.nodos[nodoi].sigma[3]     )) 
    
    print('\n Principal stress (from average):')
    print('node    s1                  n1                s2                  n2')
    for nodoi in casos[0].dominio.nodos.keys():
        print('{:3d} {:10.4f}  [{:10.4f} {:10.4f}] ; {:10.4f}  [{:10.4f} {:10.4f}]'.format(
            nodoi,  
            caso.dominio.nodos[nodoi].s_prles[0],
            caso.dominio.nodos[nodoi].s_prles[1][0], caso.dominio.nodos[nodoi].s_prles[1][1],
            caso.dominio.nodos[nodoi].s_prles[2],
            caso.dominio.nodos[nodoi].s_prles[3][0], caso.dominio.nodos[nodoi].s_prles[3][1]  )) 
    
    
    '''
    print('\n'+'#'*50)
    print('Desplazamientos (en xy, aunque haya apoyo inclinado): ')
    for inodo, nodo in caso.dominio.nodos.items():
        print('{:3d}  {:12.4e}  {:12.4e}'.format(inodo, nodo.ux, nodo.uy))
    
    print('\n Tensiones nodales (promediadas):')
    print('nodo   sxx        syy        sxy       svm')
    for inodo, nodo in caso.dominio.nodos.items():
        print('{:3d} {:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(
            inodo, nodo.sigma[0],nodo.sigma[1],nodo.sigma[2],nodo.sigma[3]))
    '''
    
    progreso +='drawing...'
    info_strv.set(progreso)
    print('drawing...')

    salida_grafica(i_di)




def salida_grafica(i_di):
    global casos, v5

    caso=casos[i_di] # trae las tensiones nodales etc

    def pinta_lado_def(ni,xi,yi): # utilidad pa no escribir 3 veces lo mismo
        # pinta la deformada de cada lado del caso 0
        uxi, uyi = np.zeros(3), np.zeros(3)
        for i in range(3):
            uxi[i]= caso.dominio.nodos[ni[i]].ux
            uyi[i]= caso.dominio.nodos[ni[i]].uy # de caso, no de caso[0]
            x,y= caso.dominio.nodos[ni[i]].x, caso.dominio.nodos[ni[i]].y
            plt.plot(x + uxi[i]*uscal, y + uyi[i]*uscal, 'd', color='b', markersize=3)
        zona = Zona(xi+uxi*uscal, yi+uyi*uscal)
        zona.pinta_zona(colorin='b', alfalin=1, ancholin=1,  todoL=True)


    def pinta_sigmas(niveles_s, tema='terrain', apariencia=0 ):
        # hay que dar unos niveles_s[] -> ya no es opcional
        # el plt.show() no se hace aqui -> hacerlo tras la llamada
        # con apariencia=0 se usa tricontourf(), 
        # con =1 se usa tripcolor() con shadding=flat
        # con =2 se usa tripcolor() con shadding=gouraud


        plt.close(fig='sigma_xx')
        plt.figure('sigma_xx')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[0][0],niveles_s[0][1],15):
            niveles.append(nivel)
                
        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, sxx, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, sxx, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[0][0], vmax=niveles_s[0][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, sxx, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[0][0], vmax=niveles_s[0][1])
                
        cb=fig.colorbar(rellenos, label='sigma_xx')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')


        plt.close(fig='sigma_yy')
        plt.figure('sigma_yy')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[1][0],niveles_s[1][1],15):
            niveles.append(nivel)

        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, syy, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, syy, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[1][0], vmax=niveles_s[1][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, syy, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[1][0], vmax=niveles_s[1][1])

        cb=fig.colorbar(rellenos, label='sigma_yy')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')


        plt.close(fig='sigma_xy')
        plt.figure('sigma_xy')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[2][0],niveles_s[2][1],15):
            niveles.append(nivel)

        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, sxy, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, sxy, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[2][0], vmax=niveles_s[2][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, sxy, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[2][0], vmax=niveles_s[2][1])
                    
        cb=fig.colorbar(rellenos, label='sigma_xy')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')


        plt.close(fig='sigma_vm')
        plt.figure('sigma_vm')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[3][0],niveles_s[3][1],15):
            niveles.append(nivel)
                
        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, svm, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, svm, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[3][0], vmax=niveles_s[3][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, svm, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[3][0], vmax=niveles_s[3][1])
                    
        cb=fig.colorbar(rellenos, label='sigma_vm')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')



        plt.close(fig='Isostatic lines')
        plt.figure(   'Isostatic lines')
        
        # hacer cuadrangulacion para pintar isostaticas como "stream":
        xci, yci = np.meshgrid(
            np.linspace(caso.dominio.extremosx[0], caso.dominio.extremosx[1], 60),
            np.linspace(caso.dominio.extremosy[0], caso.dominio.extremosy[1], 60)  )
        
        # crear un interpolador lineal para s1:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, s1, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        s1_interpolau = interp_lin(xci, yci)
        
        # crear un interpolador lineal para n10:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, n10, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        n10_interpolau = interp_lin(xci, yci)
        
        # crear un interpolador lineal para n11:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, n11, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        n11_interpolau = interp_lin(xci, yci)
        
        # crear un interpolador lineal para s2:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, s2, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        s2_interpolau = interp_lin(xci, yci)
        
        # crear un interpolador lineal para n20:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, n20, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        n20_interpolau = interp_lin(xci, yci)
        
        # crear un interpolador lineal para n21:
        interp_lin = matplotlib.tri.LinearTriInterpolator(triangulacion, n21, trifinder=el_trifinder) 
        # calcular valores interpolados para la malla rectangular
        n21_interpolau = interp_lin(xci, yci)
        
        # Los kkk_interpolau son masked arrays que invalidan los puntos que no estan en
        # ningun triangulo (estan fuera del solido). Su mascara es igual para todos, digo yo,
        # -> la de s1_interpolau = a la de  por ej. Adicionalmente aplico una mascara que 
        # invalida los valores de t.pr. que son mayores en abs() que lo que limita el usuario.
        # Aplico la mascara compuesta (masc) a los datos de todos los kkk_interpolau.
        
        with np.errstate(invalid='ignore'): # para que no de error con los NaN
            m1= abs(s1_interpolau.data) > niveles_s[6][1]
            m2= abs(s2_interpolau.data) > niveles_s[6][1]
        
        s1_interpolau = np.ma.array(s1_interpolau, mask=m1)
        n10_interpolau= np.ma.array(n10_interpolau, mask=m1)
        n11_interpolau= np.ma.array(n11_interpolau, mask=m1)
        s2_interpolau = np.ma.array(s2_interpolau, mask=m2)
        n20_interpolau= np.ma.array(n20_interpolau, mask=m2)
        n21_interpolau= np.ma.array(n21_interpolau, mask=m2)
        
        # dibujarlo (variando grosor):
        
        a= niveles_s[6][1] 
        #lw= 5* np.ma.masked_where(abs(s1_interpolau > niveles_s[6][1]), s1_interpolau) / a
        lw= 5* s1_interpolau / a
        plt.streamplot(xci, yci,
                n10_interpolau*s1_interpolau, n11_interpolau*s1_interpolau, 
                arrowsize=0, density=1, color='b', linewidth=lw, minlength=0.2)
        
        #lw= 5* np.ma.masked_where(abs(s2_interpolau > niveles_s[6][1]), s2_interpolau) / a
        lw= 5* s2_interpolau / a
        plt.streamplot(xci, yci, 
                n20_interpolau*s2_interpolau, n21_interpolau*s2_interpolau, 
                arrowsize=0., density=1, color='r', linewidth=lw, minlength=0.2)

        for e in casos[0].dominio.elems.values():
            e.pinta_elem(colorin='grey', ancholin=1.4, alfalin=0.6)
        
        patch_azul= matplotlib.patches.Patch(color='blue', label='sigma I')
        patch_naranja= matplotlib.patches.Patch(color='r', label='sigma II')
        plt.legend(handles=[patch_azul, patch_naranja])



        plt.close(fig='Tractions (s_I)')
        plt.figure(   'Tractions (s_I)')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[4][0],niveles_s[4][1],15):
            niveles.append(nivel)
                
        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, s1, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, s1, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[4][0], vmax=niveles_s[4][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, s1, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[4][0], vmax=niveles_s[4][1])
                
        cb=fig.colorbar(rellenos, label='sigma_I pr')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')
    
    

        plt.close(fig='Compressions (s_II)')
        plt.figure(   'Compressions (s_II)')
        fig, ax = plt.gcf(), plt.gca()

        niveles=[]
        for nivel in np.linspace(niveles_s[5][0],niveles_s[5][1],15):
            niveles.append(nivel)
                
        if apariencia==0:
            rellenos= ax.tricontourf(triangulacion, s2, niveles, cmap=tema, alpha=0.5)
        elif apariencia==1:
            rellenos= ax.tripcolor(triangulacion, s2, cmap=tema, edgecolors='none',
                        shading='flat', vmin=niveles_s[5][0], vmax=niveles_s[5][1])
        elif apariencia==2:
            rellenos= ax.tripcolor(triangulacion, s2, cmap=tema, edgecolors='none',
                    shading='gouraud', vmin=niveles_s[5][0], vmax=niveles_s[5][1])
                
        cb=fig.colorbar(rellenos, label='sigma_II pr')
        for el in casos[0].dominio.elems.values(): 
            el.pinta_elem(colorin='grey', ancholin=1.4, tipolin='-')
    



        #plt.figure('Triángulos para dibujar')
        #plt.triplot(triangulacion, 'k.-', lw=0.2)
        #for el in casos[0].dominio.elems.values(): 
        #    el.pinta_elem(colorin='g', ancholin=1.4, tipolin='--')
        


    def redibujar():
        global v5
        for i in range(7):
            niveles_s[i]=[ float(strbajos[i].get()), float(straltos[i].get()) ]
        
        if di_tema.get() =='2':    tema='Greys'
        elif di_tema.get() =='1':  tema='coolwarm'
        else:                      tema='terrain'
        
        apariencia=int(di_apariencia.get())

        pinta_sigmas(niveles_s, tema, apariencia)
        plt.show(block=False)


    # dibujar desplazamientos.
    # hago los trazados sobre el caso 0, pero con los u del caso i_di:
    
    umax=0. # escala para trazar los desplazamientos
    anchox, anchoy = casos[0].dominio.anchox, casos[0].dominio.anchoy
    epsx= anchox/160
    for n in caso.dominio.nodos.values():
        if abs(n.ux)>umax: umax=abs(n.ux)
        if abs(n.uy)>umax: umax=abs(n.uy)
    uscal = 0.02*(anchox+anchoy)/umax
    
    plt.close(fig='Displacement')
    plt.figure('Displacement')
    
    for nodoi,n in casos[0].dominio.nodos.items():
        plt.plot(n.x, n.y, 'd', color='grey', markersize=3)
        texto='{:8.4f}\n{:8.4f}'.format(caso.dominio.nodos[nodoi].ux, 
                                        caso.dominio.nodos[nodoi].uy)
        plt.text( n.x+epsx, n.y, texto ,fontsize=9, va='center')
        
    for e in casos[0].dominio.elems.values():
        e.pinta_elem(colorin='grey')
                
        # la deformada del elemento:
        ni= e.nodse[0:3]
        xi, yi= e.xi[0:3], e.yi[0:3]
        pinta_lado_def(ni,xi,yi)
        
        ni= e.nodse[2:5]
        xi, yi= e.xi[2:5], e.yi[2:5]
        pinta_lado_def(ni,xi,yi)
        
        ni= [ e.nodse[4],e.nodse[5],e.nodse[0] ]
        xi, yi= [ e.xi[4],e.xi[5],e.xi[0] ] , [ e.yi[4],e.yi[5],e.yi[0] ]
        pinta_lado_def(ni,xi,yi)


    # dibujo con tensiones en texto, sobre el caso0:

    plt.close(fig='Stresses (annotated)')
    plt.figure(   'Stresses (annotated)')
    
    for nodoi,n in casos[0].dominio.nodos.items():
        plt.plot(n.x, n.y, 'd', color='m', markersize=3)
        sigma= caso.dominio.nodos[nodoi].sigma
        texto='{:8.4f}\n{:8.4f}\n{:8.4f}\n{:8.4f}'.format(sigma[0], 
                            sigma[1], sigma[2], sigma[3])
        plt.text( n.x+epsx, n.y, texto ,fontsize=9, va='center')

    for e in casos[0].dominio.elems.values():
        e.pinta_elem(colorin='m', ancholin=1.4, alfalin=0.6)

    plt.annotate('s_xx\ns_yy\ns_xy\ns_vm', xy=(0.95,0.03), xycoords='axes fraction', 
                 fontsize=9, color='b', fontweight='bold')


    # dibujo de tens. prles. con texto, sobre caso o caso0:
    
    plt.close(fig='Principal stresses ')
    plt.figure('Principal stresses')
    
    eps_tpr = (anchox+anchoy)/130  # el epsx no quedaba bien
    if len(casos) >=2 and len(casos[0].dominio.elems)<8:
        caso_ref = casos[1] 
    else: 
        caso_ref = casos[0]

    for nodoi, n in caso_ref.dominio.nodos.items():
        a= caso.dominio.nodos[nodoi].s_prles
        s1,n1,s2,n2 = a[0], a[1], a[2], a[3]
        centro = np.array([n.x, n.y])
        # la dir pr 1:
        p = centro - eps_tpr*n1
        q = centro + eps_tpr*n1
        plt.plot( (p[0],q[0]), (p[1],q[1]), color='b', linewidth=1.5)
        # la dir pr 2:
        p = centro - eps_tpr*n2
        q = centro + eps_tpr*n2
        plt.plot( (p[0],q[0]), (p[1],q[1]), color='r', linewidth=1.5)
        # texto con los valores :
        texto='{:8.4f}\n{:8.4f}'.format(s1, s2)
        plt.text( n.x+epsx, n.y, texto ,fontsize=9, va='center')

    for e in casos[0].dominio.elems.values():  # el dibujo base es caso0 siempre
        e.pinta_elem(colorin='grey', ancholin=1.4, alfalin=0.6)

    patch_azul= matplotlib.patches.Patch(color='blue', label='sigma I')
    patch_naranja= matplotlib.patches.Patch(color='r', label='sigma II')
    plt.legend(handles=[patch_azul, patch_naranja])


    '''
    # quizas no queramos colorines:
    seguimos= messagebox.askyesno(message='¿Quieres dibujos en colorines de las tensiones?',
        icon='question', default='no', title='Pregunta:')

    if not seguimos:
        plt.show()
        return()
    '''


    # Dibujar tensiones (2a parte, las con opcion a repintarse).
    # Hemos quedado en que las u son parabolicas -> las sigma lineales, y por
    # eso es bobada andar con ptos de gaus & interpolar plano etc, porque 
    # todo, incluido el caculo en nodos, va a estar en el mismo plano. Aqui
    # tengo ya en los nodos las sigma promediadas, que es lo que usare porque
    # es lo mejor que hay. Puede ser que un tipo de dibujo de pegotes quede
    # mejor...
    
    # generamos triangulacion 1 elem -> 4 triang y asignamos valores.
    # Para controlar la posicion en los arrays uso i
    
    xi, yi, sxx, syy, sxy, svm = [],[],[],[],[],[]
    s1, n10, n11, s2, n20, n21 = [],[],[],  [],[],[]
    renum, triangulos = {}, []
    
    i=0
    for nodoi, nodo in caso.dominio.nodos.items():
        xi.append(nodo.x)
        yi.append(nodo.y)
        renum[nodoi]=i
        i += 1
        # aprovechamos para generar las listas de sigmas diversos:
        sxx.append(nodo.sigma[0])
        syy.append(nodo.sigma[1])
        sxy.append(nodo.sigma[2])
        svm.append(nodo.sigma[3])
        
        s1.append(nodo.s_prles[0])
        n10.append(nodo.s_prles[1][0])
        n11.append(nodo.s_prles[1][1])
        s2.append(nodo.s_prles[2])
        n20.append(nodo.s_prles[3][0])
        n21.append(nodo.s_prles[3][1])
        
        
    for el in caso.dominio.elems.values():
        triangulos.append( [renum[el.nodse[0]], renum[el.nodse[1]], renum[el.nodse[5]]] )
        triangulos.append( [renum[el.nodse[2]], renum[el.nodse[3]], renum[el.nodse[1]]] )
        triangulos.append( [renum[el.nodse[3]], renum[el.nodse[5]], renum[el.nodse[1]]] )
        triangulos.append( [renum[el.nodse[4]], renum[el.nodse[5]], renum[el.nodse[3]]] )
    
    triangulacion = matplotlib.tri.Triangulation(xi, yi, triangulos)
    el_trifinder  = triangulacion.get_trifinder()
    
    
    # asignar s_extremos & inicializar niveles_s
    s_extremos=[]
    s_extremos.append([min(sxx) , max(sxx)])
    s_extremos.append([min(syy) , max(syy)])
    s_extremos.append([min(sxy) , max(sxy)])
    s_extremos.append([min(svm) , max(svm)])
    s_extremos.append([min(s1)  , max(s1)])
    s_extremos.append([min(s2)  , max(s2)])
    s_extremos.append([ 0., max( abs(max(s1)), abs(min(s1)), abs(max(s2)), abs(min(s2)) )])
    
    niveles_s=[]
    for i in range(7):
        niveles_s.append(s_extremos[i])


    # dibujo inicial de tensiones:
    try:
        pinta_sigmas(niveles_s)
    except ValueError:
        texto= 'Error when plotting stresses:'
        texto2= str(sys.exc_info()[0])
        messagebox.showinfo(message=texto,detail=texto2, title='Error',parent=v0)

    plt.show(block=False)



    # dibujos personalizados a traves de gui v5:

    v5 = Toplevel(v0)
    v5.title('Customization of graphics')

    frame_general = ttk.Frame(v5, padding='4')
    frame_general.grid(sticky=(N, W, E, S))

    frame_limites= ttk.Labelframe(frame_general, text='Limit values',
        style='jc.TLabelframe')
    frame_limites.grid(column=0,row=0, columnspan=2,ipadx=6, ipady=6)

    frame_colores= ttk.Labelframe(frame_general, text='Color scheme',
        style='jc.TLabelframe')
    frame_colores.grid(column=0,row=1, ipadx=6, ipady=6)

    frame_apariencia= ttk.Labelframe(frame_general, text='Appearance',
        style='jc.TLabelframe')
    frame_apariencia.grid(column=1,row=1,ipadx=6, ipady=6)
    
    boton_redibujar=ttk.Button(v5, text='re-draw', command= redibujar)
    boton_redibujar.grid(column=0, row=1, pady=6)

    # rellenar frame_limites
    
    texto=['s_xx','s_yy','s_xy','s_vm','s_I','s_II' ]
    for i in range(6):
        ttk.Label(frame_limites,text=texto[i]).grid(
                        row=0, column=2*i, columnspan=2, padx=3, pady=0)
    ttk.Label(frame_limites,text='s_pr').grid(row=0, column=12, padx=3, pady=0)    

    for i in range(6):
        a=ttk.Label(frame_limites,text='min max')
        a.grid(row=1, column=2*i, columnspan=2, padx=1, pady=0)
    ttk.Label(frame_limites,text='|max|').grid(row=1, column=12, padx=3, pady=0)

    entr_bajos, entr_altos = [], []
    slidebajos, slidealtos= [], []
    strbajos, straltos = [] , []

    for i in range(6): 

        # slides para todos con su StrVar salvo s_pr
        strbajos.append(StringVar(value=s_extremos[i][0]))
        straltos.append(StringVar(value=s_extremos[i][1]))
        
        slidebajo= ttk.Scale(frame_limites, orient='vertical', length=200, 
                   from_=s_extremos[i][1], to=s_extremos[i][0], variable=strbajos[i])
        slidebajos.append(slidebajo)

        slidealto= ttk.Scale(frame_limites, orient='vertical', length=200, 
                   from_=s_extremos[i][1], to=s_extremos[i][0], variable=straltos[i])
        slidealtos.append(slidealto)

        # entrys altos y bajos para todos salvo s_pr
        entr_bajos.append(ttk.Entry(frame_limites, width=7, textvariable= strbajos[i]))
        entr_altos.append(ttk.Entry(frame_limites, width=7, textvariable= straltos[i]))

        # grid de todo
        entr_bajos[i].grid(row=4, column=2*i, columnspan=2, padx=4)
        entr_altos[i].grid(row=2, column=2*i, columnspan=2, padx=4)
        slidebajos[i].grid(row=3, column=2*i,   sticky='e')
        slidealtos[i].grid(row=3, column=2*i+1, sticky='w')

    # slide para s_pr:

    straltos.append(StringVar(value=str(s_extremos[6][1])))
    strbajos.append(StringVar(value='0.00'))
    slide_pr= ttk.Scale(frame_limites, orient='vertical', length=200, 
                   from_=s_extremos[6][1], to=0, variable=straltos[6])
    entralto_pr= ttk.Entry(frame_limites, width=7, textvariable= straltos[6])
    entrbajo_pr= ttk.Label(frame_limites, width=7, text='0.00')
    slidealtos.append(slide_pr)
    entr_altos.append(entralto_pr)
    entr_bajos.append(entrbajo_pr)
    
    entr_altos[6].grid(row=2, column=12, padx=4)
    entr_bajos[6].grid(row=4, column=12, padx=4)
    slidealtos[6].grid(row=3, column=12)
    
    
    # rellenar frame_colores
    
    botones_tema, di_tema = [], StringVar()
    texto=['terrain','cold-hot','grey_scale']
    for i in range(3):
        botonico=Radiobutton(frame_colores,variable=di_tema,value=str(i),
                    background='#D9D9D9')
        botonico.grid(row=i, column=0, padx=4, pady=4) 
        botones_tema.append(botonico)
        ttk.Label(frame_colores, text=texto[i]).grid(row=i, column=1, sticky='w')
    di_tema.set('0')
    
    
    # rellenar frame_apariencia

    botones_apariencia, di_apariencia = [], StringVar()
    texto=['boundaries','triangles','blurred']
    for i in range(3):
        botonico=Radiobutton(frame_apariencia,variable=di_apariencia,value=str(i),
                    background='#D9D9D9')
        botonico.grid(row=i, column=0, padx=4, pady=4) 
        botones_apariencia.append(botonico)
        ttk.Label(frame_apariencia, text=texto[i]).grid(row=i, column=1, sticky='w')
    di_apariencia.set('0')

    v5.mainloop()



def calcula():
    global casos, i_di_strv
    
    def proceder():
        global casos, i_di_strv
        i_di= int(i_di_strv.get())
        cierraplots()
        v1.destroy()
        if casos[i_di].calculado:
            info_strv.set('drawing (case is already calculated)')
            salida_grafica(i_di)
            return(0)
        motor_calculo(i_di) 
    
    if len(casos)==0:
        info_strv.set('There is no data for calculation.')
        return(1)
    
    v1= Toplevel(v0)
    v1.title('Calculate')
    ttk.Label(v1, text='Choose a mesh to calculate:', font=('bold',14),
            background='#EDECEB').grid(row=0, column=0, columnspan=2, padx=8, pady=5)
    
    botonicos = []
    for i in range(len(casos)):
        nnodos, nelems = len(casos[i].dominio.nodos), len(casos[i].dominio.elems)
        texto = f'Mesh {i},  {nnodos} nodes '
        if casos[i].calculado: texto += '(calculated)'
        ttk.Label(v1, text=texto,background='#EDECEB').grid(
                    row=i+1, column=1, padx=8, pady=3, sticky='w')
        
        botonico=Radiobutton(v1,variable=i_di_strv,value=str(i),background='#EDECEB')
        botonico.grid(row=i+1, column=0, padx=4, pady=4) 
        botonicos.append(botonico)
    
    #i_di_strv.set('0')
    ttk.Separator(v1,orient=HORIZONTAL).grid(row=len(casos)+2, column=0,
                                            columnspan=2, pady=15, sticky='ew')
    
    boton_proceder=ttk.Button(v1, text='proceed' , command= proceder)
    boton_proceder.grid(column=0, row=len(casos)+3, 
                        columnspan=2,padx=3, pady=5)






def presenta_elige():
    global elige,n_f, nfcompleto


    ##### Ventana hija para licencia #####

    def licencia():
        v1=Toplevel(v0)

        texto='''This program is Free Software under the "GNU General Public License", version 2 or (at your choice) any posterior version. Basically, you can:

* Freely use the program
* Make copies of it and freely distribute them
* Study the code to find out how it works
* Modify or improve the program 

Under the following conditions:  

* If the modified program is released to the public, it must be released under the same license. This way the software can improve with new contributions, as it is done in science..
* Modifications and derived works in general must acknowledge the  original author (you cannot introduce yourself as the author).
* In this case you must mention the original author as: Juan Carlos del Caño, assistant professor at Escuela de Ingenierías Industriales, University of Valladolid (Spain)  

This program is distributed in the hope of being useful, but WITHOUT ANY WARRANTY. Please read the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  Or you can check the online version of the license at 
#  https://www.gnu.org/licenses/licenses.html
'''
        ttk.Label(v1, text='Terms of license', background='#EDECEB',
            font=('', 16)).grid(row=0, column=0, columnspan=2, pady=5)

        tcaja = Text(v1, width=45, height=30,wrap='word', font=('Sans',9),
            background='#EDECEB', foreground='green', border=None, padx=20, pady=12)
        tcaja.grid(column=0, row=1, padx=8, sticky=(N,W,E,S))
        tcaja.insert('1.0',texto)

        scb = ttk.Scrollbar(v1, orient=VERTICAL, command=tcaja.yview)
        scb.grid(column=1, row=1, sticky='ns')
        
        tcaja['yscrollcommand'] = scb.set

        ttk.Button(v1, text='got it', width=9, command=v1.destroy).grid(
            column=0, row=2, pady=4, columnspan=2)

        tcaja['state']='disabled'

        v1.grid_columnconfigure(0, weight=1)
        v1.grid_rowconfigure(0, weight=1)
        v1.grid_rowconfigure(1, weight=4)
        v1.grid_rowconfigure(2, weight=1)
        v1.geometry('+240+60')

        v1.focus()
        v1.mainloop()


    ##### Ventana hija para breviario #####

    def breviario():

        v2=Toplevel(v0)
            
        texto='''- What is this program useful for:
The ideal user would have basic knowledge of the Finite Element Method as usually applied to Linear Elasticity problems, or he/she is studying that kind of subject. The program performs a finite element approximation to a bidimensional elastic linear problem (plane stress or plane strain). Concentrated and distributed boundary conditions are admitted, including punctual springs and elastic foundations. Domain loads (volume forces and thermal load) are also admitted.
The program calculates the stress and displacement fields, then offers a set of figures and numerical results that should be useful in any solid mechanics analysis. 

This software has been built with educational purposes in mind. That said, the size of the discretization is only limited by the amount of RAM available. For your records, a problem with more than 115000 unknowns uses 0.9 Gb of RAM and the analysis takes 2 min on my computer (a Pentium 5 desktop).


- How to use it:

Six-noded triangular isoparametric finite elements are used in the approximation. Thus, regarding geometry, parabolic piecewise approximation will be obtained for the boundaries. The nodes of the element are to be given in counter-clockwise order, starting by a vertex.

It is advisable to not distort elements too much from their canonical shape (an equilateral triangle), and also that mid-side nodes lie not too far from halfway in the sides. Also, you'll find it convenient to use only the minimum number of elements that suffice to approximate the geometry and boundary conditions. There is a function "refine" that easily allows refining the mesh to the desired level afterwards.

Elements, nodes and material data are specified in the "domain" tab. Then you should visit the "bearing" and "loads" tabs to complete the definition of the problem. The data as a whole must represent a correctly defined problem in the sense of Kirchoff's unicity theorem.

The "base view" button shows graphically the data contained in the user interface. After refining the discretization to the desired level (clicking the "refine" button one or more times), you may click the "calculate" button to perform the FEM analysis. Among other windows, one appears whose title is "customization of graphics". Here you can adjust several plotting parameters.

When hovering the mouse over the interface, some "tooltips" do emerge. Those provide relevant info to use the program to its full, and in fact they aim to be "the user manual" (along with these notes). You may want to run the provided example as a quick way to get familiar with the intended workflow, and then try your own changes etc


- Units:

The user is expected to give all data in a coherent unit system of his/her choice. The program does not question that. The results will be presented in that same unit system. In that follows, the unit for distance, force and temperature will be noted L, F, ºT, respectively. The unit for angle, where required, is the sexagesimal degree in all cases. 

Calculations are made per unit thickness. The field "thickness" in the domain tab is only a reminder (appear as 1.0 and cannot be changed). The assumed data units are as follows:
-- x,y, nodal coordinates: L
-- Young modulus: F / L^2
-- Poisson coefficient: (non dimensional)
-- Thermal expansion coefficient: 1 / ºT
-- prescribed displacement: L (please type "free" for a free component)
-- punctual spring: F / L^2 (=F/(despl*thickness))
-- distributed spring: F / L^3 (=F/(despl*thickness*boundary_line))
-- distributed boundary load: F / L^2 (=F/(thickness*boundary_line))
-- volume force: F/ L^3
-- thermal load (temperature): ºT
-- punctual force: F / L (=F/(thickness))

As an example, let N & mm be our choice for force and distance units. Our solid is 2mm thick. A spring with a stiffness of 100N/mm in the real world must be passed to the program as 50 (N/mm^2). A concentrated force of physically 90N must be passed to the program as 45 (N/mm). Distributed loads (domain or boundary type) are to be passed with their nominal units (N/mm^2 or N/mm^3). If the data for a boundary distributed load is available as per unit of length (as is often the case in flexural beams), you must divide that by the thickness in order to get the correct units.
The program will calculate values of displacement and stress aiming to be the real ones, in mm & N/mm^2 respectively. The "reactive forces" (at supports etc) will be given as per unit thickness.

Rationale: The possibility of using elements of different thickness in the same discretization is often found in FEA software. This indeed involves either a "supplementary approximation" on top of those of the plane problem or a flagrant violation of the hypothesis on which plane problems are formulated, depending on how rigorous you want to be. Its inclusion has been considered unsuitable in educational software. Once decided thickness will be constant across the solid, the eventual specification of a thickness on the user side would not avoid the necessity of an explanation about conventions (as in the previous paragraph). So, calculating as per unit thickness gives in practice the same functionality and generality with one less parameter.


- The example (and related tips).

The geometry could represent a vault, or a dome's rib, or a flying buttress of a cathedral, etc. The base mesh has a single element in this case. Clearly, a stress singularity will occur at the leftmost point. It is known that strict convergence of the solution cannot be achieved in this case (at least using regular elements). The example has been chosen that way by intention as there's something to learn about it. Regarding the results:
-- The s_I plot, principal stress I, shows a traction concentration near the left lower point, which does not seem to be associated with the singularity. You can investigate more on it by adjusting colour levels in "Customization of graphics".
-- The isostatic lines are plotted with a thickness proportional to the stress value. They appear very thin in this case, the ones associated with s_I being barely noticeable. The reason for that can be seen in the "Principal stress" plot: the maximum absolute value of principal stress, 1379 is pretty meaningless as it is clearly associated with the singularity. Maximum values of about 300 can be considered more realistic in this problem. You can go to the "Customization of graphics" window and adjust "s_pr" to a maximum of 300, then hit "re-plot". You'll get a better representation of isostatic lines.
-- Please note: in order to keep things tidy on the screen, figures may be plotted out of proportionality. For this reason, the s_I & s_II families may not appear perpendicular (though they are). All it takes is to resize the window until you get equal spacing in x & y axes.
-- You are encouraged to modify loads and bearing, then "read gui" to incorporate your changes, "base view" to visualize what you've written, "refine" the mesh, "calculate"... and inadvertently you'll find yourself thinking on the problem, not the interface. That's the intent anyway.


- Current state and roadmap:

jMef aims to be useful as a daily tool for students and teacher in an introductory course on the Theory of Elasticity. This version 0.6 is the first one to be released (2021). It includes all the foreseen functionality regarding its goals. 

The memory requirements are moderate and the execution time is pretty decent. The use of some precompiled routines available in Scipy Python's library has been critical for that. The code could still take advantage of some review regarding optimization, although in the author's feeling it's dubious whether its impact on the overall performance would be noticeable or not. 

There are plans to include axisymmetric problems and Saint-Venant's torsion in the future, as well as thermal conduction. Some further help to establish the "base mesh" is also in the list. There are no plans to take the program out of the linear isotropic bidimensional scope though.

The author will be thankful for any report on bugs in the program.

I hope you'll find jMef useful.
JC del Caño
____________________________________
'''
        
        ttk.Label(v2, text='Quick notes', background='#EDECEB',
            font=('', 16)).grid(row=0, column=0, columnspan=2, pady=5)

        tcaja = Text(v2, width=60, height=35,wrap='word', font=('Sans',9),
            background='#EDECEB', foreground='green', border=None, padx=20, pady=12)
        tcaja.grid(column=0, row=1, padx=8, sticky=(N,W,E,S))
        tcaja.insert('1.0',texto)

        scb = ttk.Scrollbar(v2, orient=VERTICAL, command=tcaja.yview)
        scb.grid(column=1, row=1, sticky='ns')
        
        tcaja['yscrollcommand'] = scb.set

        ttk.Button(v2, text='got it', width=9, command=v2.destroy).grid(
            column=0, row=2, pady=4, columnspan=2)

        tcaja['state']='disabled'

        v2.grid_columnconfigure(0, weight=1)
        v2.grid_rowconfigure(0, weight=1)
        v2.grid_rowconfigure(1, weight=4)
        v2.grid_rowconfigure(2, weight=1)
        v2.geometry('+250+70')

        v2.focus()
        v2.mainloop()



    ##### Para el prb por defecto #####

    def default_prb():
        global elige
        elige='default' # sin mas ventanas
        v0.destroy()



    ##### La ventana de inicio (por fin) #####

    v0.title("jMef v0.6")

    cuadro = ttk.Frame(v0, padding='9 3 3 3') 
    cuadro.grid(column=0, row=0, sticky=(N, W, E, S))

    ttk.Label(cuadro, text='jMef v0.6', font=('', 40)).grid(row=0,
        column=0, columnspan=4)
    ttk.Label(cuadro, text='Finite Element Analysis - 2D elasticity', 
        font=('Courier', 16)).grid(row=1, column=0, columnspan=4)
    ttk.Label(cuadro, text='by:   Juan Carlos del Caño\n').grid(row=2,
        column=0, columnspan=4)

    # hago la parte izda de la ventana

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(column=0, row=3,
        columnspan=4, sticky='ew')
        
    texto= 'This is Free Software. That brings to you some rights\n'
    texto +='and also some obligations. Please read the license.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=4, 
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=5, column=0,
        columnspan=3, sticky='ew')
        
    texto=  "If you don't know what this program is for or you\n"
    texto +='have a basic question about it, please read these quick\n'
    texto +='notes. Then load the example and experiment on it.\n'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=6, 
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=7, column=0,
        columnspan=3, sticky='ew')
        
    texto = 'You can start a new problem from scratch.\n'
    texto +='You will provide geometry (nodes & elements) and\n'
    texto +='other general data. You can also load a previously\n'
    texto +='saved problem.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=8,
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro,orient=HORIZONTAL).grid(row=9,column=0,sticky='ew')
    ttk.Separator(cuadro,orient=HORIZONTAL).grid(row=9,column=2,sticky='ew')
    ttk.Label(cuadro, text='or:').grid(row=9,column=1)
        
    texto = 'You can load a hardcoded example. It shows some\n'
    texto +='of the capabilities of the program. You can play\n'
    texto +='around with it or use it as a quick check on the\n'
    texto +='correct installation of jMef & dependencies.'

    ttk.Label(cuadro, text=texto, foreground='green').grid(row=10,
        column=0, columnspan=3, sticky='w')
        
    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=11, column=0,
        columnspan=4, sticky='ew')


    # ahora hago la parte derecha

    ttk.Separator(cuadro,orient=VERTICAL).grid(row=3,column=3,
        rowspan=9, sticky='ns')
    ttk.Button(cuadro, text='License',
        command=licencia).grid(row=4, column=3)

    ttk.Button(cuadro, text='Quick notes', command=breviario).grid(
        row=6, column=3)

    ttk.Button(cuadro, text='Solve', command=v0.destroy).grid(
        row=8, column=3)

    ttk.Button(cuadro, text='Example', command=default_prb).grid(
        row=10, column=3)

    for hijo in cuadro.winfo_children():
        hijo.grid_configure(padx=12, pady=8)

    v0.geometry('+70-70')
    v0.focus()
    v0.mainloop()
    
    return()

###################### Fin de ventana de inicio #####################





#####################################################################
#####################################################################
####                    PROGRAMA PRINCIPAL                       ####
####  Es la interfaz GUI de control y algunas inicializaciones   ####
#####################################################################
#####################################################################


elige, n_f, nfcompleto = '', '', ''
casos, v5 = [], None

v0=Tk()
presenta_elige() # v0 es auto destruida 



v0=Tk()
v0.title('jMef v0.6')

estilo = ttk.Style()
estilo.configure('jc.TLabelframe.Label', foreground ='green')
estilo.configure('jc_red.TButton',foreground='#9B2803')
estilo.configure('jc.TCheckbutton', background='#EDECEB')
i_di_strv =  StringVar() # hago esto para poder hacerlo global y que recuerde la ultima
                         # malla calculada. Aqui pq tiene que estar el Tk() activo

frame_general = ttk.Frame(v0, padding='4')
frame_general.grid(sticky=(N, W, E, S))

frame_acciones= ttk.Labelframe(frame_general, text='Control',
    style='jc.TLabelframe', width=170,height=550)
frame_acciones.grid(column=0,row=0,ipadx=3, sticky='s')

frame_pestanas= ttk.Notebook(frame_general)
frame_pestanas.grid(column=1, row=0, columnspan=3)

info_strv = StringVar()
info_strv.set('(no recent events)')
info_lbl= ttk.Label(frame_general, width=82, textvariable=info_strv, foreground='green')
             # ,borderwidth=2, relief="groove", background='#EDECEB')
info_lbl.grid(column=0, row=1, padx=10, sticky='w', columnspan=2)
CreateToolTip(info_lbl,'informative zone - recent events.')

ttk.Label(frame_general, text='Meshes:', foreground='green').grid(column=2, row=1, sticky='e')
ncasos_strv = StringVar()
ncasos_strv.set('0')
ncasos_lbl= ttk.Label(frame_general, width=3, textvariable=ncasos_strv,
            foreground='green') #, background='#EDECEB')
ncasos_lbl.grid(column=3, row=1, sticky='e')

frame_dominio=    ttk.Frame(frame_pestanas)
frame_sustentacion  =   ttk.Frame(frame_pestanas)
frame_cargas=   ttk.Frame(frame_pestanas)

frame_pestanas.add(frame_dominio, text='Domain')
frame_pestanas.add(frame_sustentacion, text='Bearing')
frame_pestanas.add(frame_cargas, text='Loads')


# botones de acciones
boton_cargar=ttk.Button(frame_acciones, text='load', command= a_cargar)
boton_cargar.grid(column=0, row=0, padx=3, pady=(20,5))
CreateToolTip(boton_cargar,'loads a previously saved problem')

boton_guardar=ttk.Button(frame_acciones, text='save', command=a_guardar)
boton_guardar.grid(column=0, row=1, padx=3, pady=5)
CreateToolTip(boton_guardar,"saves current gui's data") 

ttk.Separator(frame_acciones,orient=HORIZONTAL).grid(
    row=2,column=0,pady=20, sticky='ew')

boton_salir=ttk.Button(frame_acciones, text='-quit-', style='jc_red.TButton',command=a_salir)
boton_salir.grid(column=0, row=3, padx=3, pady=5)

boton_especial=ttk.Button(frame_acciones, text='special', 
style='jc_red.TButton' , command=especial)
boton_especial.grid(column=0,row=4, padx=3, pady=5)
CreateToolTip(boton_especial,'view a refined mesh and/or\nmake it the "base mesh"')

boton_limpiar=ttk.Button(frame_acciones,text=' clean', style='jc_red.TButton', command= a_limpiar)
boton_limpiar.grid(column=0, row=5, padx=3, pady=5)
CreateToolTip(boton_limpiar,'utility to erase all\nfields in a tab')

ttk.Separator(frame_acciones,orient=HORIZONTAL).grid(
    row=6,column=0,pady=20, sticky='ew')

boton_actualizar =ttk.Button(frame_acciones,text='read gui', command=leer_gui)
boton_actualizar.grid(column=0, row=7, padx=3, pady=5)
CreateToolTip(boton_actualizar,'updates the changes you\n  manually introduced')

boton_vistazo =ttk.Button(frame_acciones,text='base view', command=pinta_vistazo)
            #command= lambda: pinta_vistazo(caso)) 
boton_vistazo.grid(column=0, row=8, padx=3, pady=5)
CreateToolTip(boton_vistazo,'shows the base case\n(data in the interface)')

boton_refinar=ttk.Button(frame_acciones, text='refine', command= refinar4)
boton_refinar.grid(column=0, row=9, padx=3, pady=5)
texto = 'refines the mesh by\ndividing in four each element'
CreateToolTip(boton_refinar, texto)

boton_calcula=ttk.Button(frame_acciones,text='calculate', command= calcula)
boton_calcula.grid(column=0, row=10, padx=3, pady=5)
CreateToolTip(boton_calcula,'run the analisys')

ttk.Separator(frame_acciones,orient=HORIZONTAL).grid(
    row=11,column=0, pady=(15,10), sticky='ew')

boton_cerrargr=ttk.Button(frame_acciones,text='   close\ngraphics',command= cierraplots)
boton_cerrargr.grid(column=0, row=12, padx=3, pady=(5,10))


############################################
# rellenar la ventana-pestana frame_dominio
############################################

frame_nodos=ttk.Labelframe(frame_dominio, text='Nodes', style='jc.TLabelframe')
frame_elems=ttk.Labelframe(frame_dominio, text='Elements', style='jc.TLabelframe')
frame_ctes= ttk.Labelframe(frame_dominio, text='Materials', style='jc.TLabelframe')
frame_tdpe= ttk.Frame(frame_dominio)

frame_nodos.grid(row=0, column=0, rowspan=3, sticky='n', padx=10, pady=10)
frame_elems.grid(row=0, column=1, sticky='n', padx=(0,10), pady=10)
frame_ctes.grid(row=1,column=1, sticky='n', padx=(0,10), pady=6)
frame_tdpe.grid(row=2, column=1, sticky='new', padx=(0,10), pady=6)

texto=''' please write the data separated by
spaces (as many as desired) in successive rows
(units: length)'''
CreateToolTip(frame_nodos,texto)
texto='''element code, material code (matching the
table below), and codes of nodes (anti clockwise,
starting by a vertex).'''
CreateToolTip(frame_elems,texto)
texto='''code of material, and elastic constants
(units: pressure, non-dimensional, Tdegree^(-1)'''
CreateToolTip(frame_ctes, texto)

    # subventana frame_dominio -> frame_nodos 

texto=' nº       x        y'
ttk.Label(frame_nodos, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_nodos = Text(frame_nodos, width=30, height=28)
wgt_nodos.grid(row=1, column=0, sticky='nsew')

scb_n = ttk.Scrollbar(frame_nodos, orient=VERTICAL, command=wgt_nodos.yview)
scb_n.grid(column=1, row=1, sticky='ns')
wgt_nodos['yscrollcommand'] = scb_n.set


    # subventana frame_dominio -> frame_elems

texto=' nº  mat      <--- 6 nodes --->'
ttk.Label(frame_elems, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_elems = Text(frame_elems, width=40, height=14)
wgt_elems.grid(row=1, column=0, sticky='nsew')

scb_e = ttk.Scrollbar(frame_elems, orient=VERTICAL, command=wgt_elems.yview)
scb_e.grid(column=1, row=1, sticky='ns')
wgt_elems['yscrollcommand'] = scb_e.set


    # subventana frame_dominio -> frame_ctes

texto=' nº   Young    Poiss   alfaT'
ttk.Label(frame_ctes, text=texto, font=('mono',10) ).grid(
                            row=0, column=0, columnspan=2, sticky='w')
wgt_ctes = Text(frame_ctes, width=40, height=6)
wgt_ctes.grid(row=1, column=0, sticky='nsew')

scb_c = ttk.Scrollbar(frame_ctes, orient=VERTICAL, command=wgt_ctes.yview)
scb_c.grid(column=1, row=1, sticky='ns')
wgt_ctes['yscrollcommand'] = scb_c.set


    # subventana frame_dominio -> frame_tdpe

esp_strv=StringVar(value='1.0') # a no cambiar
ttk.Label(frame_tdpe, text= 'Thickness (remember):').grid(column=0, row=0,padx=20, sticky='sw')
ttk.Entry(frame_tdpe, textvariable=esp_strv, width=9, state='disabled').grid(
            column=0, row=1, padx=(40,25), sticky='nw')

ttk.Label(frame_tdpe,text='Plane stress').grid(
        column=1, row=0, padx=3, sticky='se')
ttk.Label(frame_tdpe,text='Plane strain').grid(
        column=1, row=1, padx=3, sticky='ne')

tplana_strv=StringVar()
tplana_strv.set('1')
t_plana=Radiobutton(frame_tdpe, variable=tplana_strv, value='1',background='#D9D9D9') 
d_plana=Radiobutton(frame_tdpe, variable=tplana_strv, value='0',background='#D9D9D9')  
t_plana.grid(column=2, row=0, sticky='sw', padx=6, pady=0)
d_plana.grid(column=2, row=1, sticky='nw', padx=6, pady=(2,10))



###################################################
# rellenar la ventana-pestana de frame_sustentacion
###################################################

frame_nodos_imp=ttk.Labelframe(frame_sustentacion, text='Prescribed-u nodes', 
            style='jc.TLabelframe')
frame_resortes_p=ttk.Labelframe(frame_sustentacion, text='Punctual springs', 
            style='jc.TLabelframe')
frame_zonas_imp=ttk.Labelframe(frame_sustentacion, text='Prescribed-u Zones', 
            style='jc.TLabelframe')
frame_resortes_s=ttk.Labelframe(frame_sustentacion, text='Elastic-bearing Zones', 
            style='jc.TLabelframe')

frame_nodos_imp.grid(row=0, column=0, padx=(6,6), pady=(16,6))
frame_resortes_p.grid(row=0, column=1, padx=(6,6), pady=(16,6))
frame_zonas_imp.grid(row=1,column=0, padx=(6,6), pady=(12,6), columnspan=2)
frame_resortes_s.grid(row=2,column=0, padx=(6,6), pady=(12,6), columnspan=2)

tex00='''- The "dir" field admits as values: "xy", "nt" (meaning normal & 
  tangencial components), or an angle in degrees from the x axis.
- In all cases an axis "1" is defined. It is x, n, or the given direction.
- Axis "2" is such as the turn of 1 over 2 is counter clockwise. It is
  coincident with "y" or "t" in the first and secon cases respectively.
- Parameters in a line are understood to be given in 1-2 axes.
'''
CreateToolTip(frame_nodos_imp,tex00+'\n (units:  length)')
CreateToolTip(frame_resortes_p,tex00+'\n (units:  stifness/thickness ~ pressure)')

tex01='''- Regarding the "dir" field -> see tip in previous windows.
- A "zone consists of three contiguous boundary nodes, nA nB nC, which
  in turn make up for an element side. They must be given counter-clockwise starting by a vertex.
- The remaining parameters are again understood in 1,2 axes'''
CreateToolTip(frame_zonas_imp, tex01+'\n ( units:  length)')
CreateToolTip(frame_resortes_s, tex01+'\n ( units:  stifness/area ~ force/volume)')



    # subventana frame_sustentacion -> frame_nodos_imp

texto='dir node    u_1     u_2'
ttk.Label(frame_nodos_imp, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_nodos_imp = Text(frame_nodos_imp, width=30, height=8)
wgt_nodos_imp.grid(row=1, column=0, sticky='nsew')

scb_n_imp = ttk.Scrollbar(frame_nodos_imp, orient=VERTICAL, command=wgt_nodos_imp.yview)
scb_n_imp.grid(column=1, row=1, sticky='ns')
wgt_nodos_imp['yscrollcommand'] = scb_n_imp.set

    # subventana frame_sustentacion -> frame_resortes_p

texto='dir node    k_1     k_2'
ttk.Label(frame_resortes_p, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_r_p = Text(frame_resortes_p, width=30, height=8)
wgt_r_p.grid(row=1, column=0, sticky='nsew')

scb_r_p = ttk.Scrollbar(frame_resortes_p, orient=VERTICAL, command=wgt_r_p.yview)
scb_r_p.grid(column=1, row=1, sticky='ns')
wgt_r_p['yscrollcommand'] = scb_r_p.set

    # subventana frame_sustentacion -> frame_zonas_imp

texto='dir  nA  nB  nC    u_1(A)   u_1(B)   u_1(C)   u_2(A)   u_2(B)   u_2(C)'
ttk.Label(frame_zonas_imp, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_z_i = Text(frame_zonas_imp, width=80, height=6)
wgt_z_i.grid(row=1, column=0, sticky='nsew')

scb_z_i = ttk.Scrollbar(frame_zonas_imp, orient=VERTICAL, command=wgt_z_i.yview)
scb_z_i.grid(column=1, row=1, sticky='ns')
wgt_z_i['yscrollcommand'] = scb_z_i.set

    # subventana frame_sustentacion -> frame_resortes_s

texto='dir  nA  nB  nC    k_1(A)   k_1(B)   k_1(C)   k_2(A)   k_2(B)   k_2(C)'
ttk.Label(frame_resortes_s, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_r_s = Text(frame_resortes_s, width=80, height=6)
wgt_r_s.grid(row=1, column=0, sticky='nsew')

scb_r_s = ttk.Scrollbar(frame_resortes_s, orient=VERTICAL, command=wgt_r_s.yview)
scb_r_s.grid(column=1, row=1, sticky='ns')
wgt_r_s['yscrollcommand'] = scb_r_s.set



##############################################
# rellenar la ventana-pestana de frame_cargas
##############################################


frame_xraya=ttk.Labelframe(frame_cargas, text='Boundary distributed loads', 
            style='jc.TLabelframe')
frame_xgrande= ttk.Labelframe(frame_cargas, text='Volume forces', style='jc.TLabelframe')
frame_temperatura= ttk.Labelframe(frame_cargas, text='Thermal load',style='jc.TLabelframe')
frame_fuerzas=ttk.Labelframe(frame_cargas, text='Punctual loads', style='jc.TLabelframe')

frame_xraya.grid(row=0, column=0, padx=(6,6), pady=(16,6), sticky='w', columnspan=2)
frame_xgrande.grid(row=1,column=0, padx=(10,0), pady=16, sticky='w', columnspan=2)
frame_temperatura.grid(row=2,column=0, padx=(10,0), pady=16, sticky='w',columnspan=2)
frame_fuerzas.grid(row=3, column=0, padx=(20,0), pady=16)

CreateToolTip(frame_xraya,tex01+'\n ( units: force / area).')
texto='Escriba funciones de x,y ( units: force / volume)'
CreateToolTip(frame_xgrande, texto)
CreateToolTip(frame_temperatura, 'Please write a function of x,y\n(units: T-degrees)')
CreateToolTip(frame_fuerzas, 'please see tips in previous frames.\n (units: force / thickness)')


    # subventana frame_cargas -> frame_xraya

texto='dir  nA  nB  nC    p_1(A)   p_1(B)   p_1(C)   p_2(A)   p_2(B)   p_2(C)'
ttk.Label(frame_xraya, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_xr = Text(frame_xraya, width=80, height=6)
wgt_xr.grid(row=1, column=0, sticky='nsew')

scb_xr = ttk.Scrollbar(frame_xraya, orient=VERTICAL, command=wgt_xr.yview)
scb_xr.grid(column=1, row=1, sticky='ns')
wgt_xr['yscrollcommand'] = scb_xr.set

    # subventana frame_cargas -> frame_xgrande

kk=ttk.Label(frame_xgrande, text='x component: ').grid(row=0, column=0)
kk=ttk.Label(frame_xgrande, text='y component: ').grid(row=1, column=0)

filas_xgrande=[]
for i in range(2):
    e=ttk.Entry(frame_xgrande, width=63)
    e.grid(row=i, column=1, padx=3)
    filas_xgrande.append(e)

    # subventana frame_cargas -> frame_temperatura

kk=ttk.Label(frame_temperatura, text='Temperature(x,y): ').grid(row=0, column=0)

filas_temperatura=[]
e=ttk.Entry(frame_temperatura, width=60)
e.grid(row=0, column=1, padx=3, pady=6)
filas_temperatura.append(e)

    # subventana frame_cargas -> frame_fuerzas

texto='dir node    F_1     F_2'
ttk.Label(frame_fuerzas, text=texto, font=('mono',10) ).grid(row=0, column=0, sticky='w')
wgt_fuerzas = Text(frame_fuerzas, width=30, height=6)
wgt_fuerzas.grid(row=1, column=0, sticky='nsew')

scb_fuerzas = ttk.Scrollbar(frame_fuerzas, orient=VERTICAL, command=wgt_fuerzas.yview)
scb_fuerzas.grid(column=1, row=1, sticky='ns')
wgt_fuerzas['yscrollcommand'] = scb_fuerzas.set

texto='    visualize domain loads\n(temperature & volume forces)'
boton_verXT= ttk.Button(frame_cargas,text=texto, command= verXT)
boton_verXT.grid(column=1, row=3, padx=3, pady=5, sticky='w') 




# el GUI esta. Si se pidio el prb de ejemplo, rellenar con sus datos y hacer
if elige=='default': 
    rellena_default()
    leer_gui()
    refinar4()
    refinar4()
    refinar4()
    motor_calculo(3)
    salida_grafica(3)
    



v0.protocol('WM_DELETE_WINDOW', a_salir)
v0.mainloop()



